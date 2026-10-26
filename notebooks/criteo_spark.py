# Databricks notebook source
# Milestone 3: Criteo CUPAC bridge (ATE on visit) in PySpark.
#
# Intended for Databricks Free Edition. If cluster memory or runtime limits
# bite, subsample with the same seed as the local path (SEED_ROOT = 20261026,
# N_sub = 500_000) and record the subsample in the run notes.
#
# This file is notebook source (`.py` with `# COMMAND ----------` cells). It does
# not commit raw Criteo data. Download the gzip once to DBFS / Volumes first.
#
# Algebraic identity (see docs/m3-writeup.md):
#   mean(Y) - lam * (mean(f_lab) - mean(f_all))
# is PPI; with f = m(X) an ML outcome model and no pre-period, that is CUPAC.

# COMMAND ----------

SEED_ROOT = 20261026
N_SUB = 500_000
CRITEO_CARD_N = 25_309_483
TFDS_TRAIN_N = 13_979_592
FEATURE_COLS = [f"f{i}" for i in range(12)]

# Set to your DBFS / Volume path after uploading the gzip from
# http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
CRITEO_PATH = "dbfs:/FileStore/criteo/criteo-research-uplift-v2.1.csv.gz"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

raw = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(CRITEO_PATH)
)
n_file = raw.count()
print(f"rows={n_file:,}; Criteo card={CRITEO_CARD_N:,}; TFDS train={TFDS_TRAIN_N:,}")

# COMMAND ----------

# Optional subsample for Free Edition limits.
use_subsample = n_file > N_SUB  # flip to False for full-file runs
if use_subsample:
    df = (
        raw.withColumn("rnd", F.rand(SEED_ROOT))
        .orderBy("rnd")
        .limit(N_SUB)
        .drop("rnd")
    )
    print(f"Using subsample n={N_SUB:,}, seed={SEED_ROOT}")
else:
    df = raw
    print("Using full file")

df = df.select(
    *[F.col(c).cast("double").alias(c) for c in FEATURE_COLS],
    F.col("treatment").cast("int").alias("treatment"),
    F.col("visit").cast("int").alias("visit"),
    F.col("conversion").cast("int").alias("conversion"),
    F.col("exposure").cast("int").alias("exposure"),
)
df.cache()
print(df.agg(
    F.mean("treatment").alias("treatment_rate"),
    F.mean("visit").alias("visit_rate"),
).collect())

# COMMAND ----------

# Difference in means ATE on visit.
stats = df.groupBy("treatment").agg(
    F.mean("visit").alias("mu"),
    F.variance("visit").alias("var"),
    F.count("*").alias("n"),
).orderBy("treatment").collect()
mu = {int(r["treatment"]): float(r["mu"]) for r in stats}
var = {int(r["treatment"]): float(r["var"]) for r in stats}
n = {int(r["treatment"]): int(r["n"]) for r in stats}
ate_dim = mu[1] - mu[0]
se_dim = (var[1] / n[1] + var[0] / n[0]) ** 0.5
print(f"DIM ATE={ate_dim:.6f} SE={se_dim:.6f}")

# COMMAND ----------

# CUPAC via Spark ML: cross-fit GBT regressor on features -> visit, then
# difference in means of Y - lam * (f - mean(f)).
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
# 5-fold cross-fit via a fold id.
df_fold = df.withColumn("fold", (F.rand(SEED_ROOT) * 5).cast("int"))
preds = None
for k in range(5):
    train = df_fold.filter(F.col("fold") != k)
    test = df_fold.filter(F.col("fold") == k)
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="visit",
        predictionCol="f_hat",
        maxDepth=5,
        maxIter=40,
        seed=SEED_ROOT + k,
    )
    model = Pipeline(stages=[assembler, gbt]).fit(train)
    part = model.transform(test).select("treatment", "visit", "f_hat")
    preds = part if preds is None else preds.unionByName(part)

pred_stats = preds.agg(
    F.mean("f_hat").alias("f_bar"),
    F.corr("visit", "f_hat").alias("corr"),
    F.variance("f_hat").alias("var_f"),
    F.variance("visit").alias("var_y"),
).collect()[0]
f_bar = float(pred_stats["f_bar"])
# lam = Cov(Y,f)/Var(f) approximated via corr * sd_y / sd_f, clipped to [0,1].
import math

sd_f = math.sqrt(max(float(pred_stats["var_f"]), 1e-12))
sd_y = math.sqrt(max(float(pred_stats["var_y"]), 1e-12))
corr = float(pred_stats["corr"] or 0.0)
lam = max(0.0, min(1.0, corr * sd_y / sd_f))
adj = preds.withColumn("y_adj", F.col("visit") - lam * (F.col("f_hat") - F.lit(f_bar)))
cupac_stats = adj.groupBy("treatment").agg(
    F.mean("y_adj").alias("mu"),
    F.variance("y_adj").alias("var"),
    F.count("*").alias("n"),
).orderBy("treatment").collect()
mu_c = {int(r["treatment"]): float(r["mu"]) for r in cupac_stats}
var_c = {int(r["treatment"]): float(r["var"]) for r in cupac_stats}
n_c = {int(r["treatment"]): int(r["n"]) for r in cupac_stats}
ate_cupac = mu_c[1] - mu_c[0]
se_cupac = (var_c[1] / n_c[1] + var_c[0] / n_c[0]) ** 0.5
var_dim = var[1] / n[1] + var[0] / n[0]
var_cupac = var_c[1] / n_c[1] + var_c[0] / n_c[0]
print(f"CUPAC ATE={ate_cupac:.6f} SE={se_cupac:.6f} lam={lam:.4f}")
print(f"variance ratio DIM/CUPAC = {var_dim / var_cupac:.4f}")

# COMMAND ----------

# A/A on control: random fake treatment, expect ~5% rejects at alpha=0.05.
ctrl = df.filter(F.col("treatment") == 0).withColumn(
    "fake_t", (F.rand(SEED_ROOT + 17) < 0.5).cast("int")
)
aa = ctrl.groupBy("fake_t").agg(
    F.mean("visit").alias("mu"),
    F.variance("visit").alias("var"),
    F.count("*").alias("n"),
).collect()
# Single split shown here; the local pandas path repeats R=200 splits.
print(aa)

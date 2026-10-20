<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" />

# Python Machine Learning Curriculum for Stock Investing

> From Python fundamentals through Transformer deep learning — a step-by-step course built around **stock-investing data analysis**.

Korean version: [readme-kr.md](readme-kr.md)

---

![alt text](image.png)

---

## <i class="fa-solid fa-folder"></i> Project structure

```
python-ml-class/
├── src/                    ← all Python lab files
│   ├── korean_font.py      ← Korean font utility (shared by every file)
│   ├── NumpyStockArray.py
│   ├── PandasPortfolio.py
│   ├── ... (23 files in total)
│
├── docs/                   ← elementary-level explanations of each source file
│   ├── NumpyStockArray.md
│   ├── PandasPortfolio.md
│   ├── ... (23 files in total)
│   └── CloudMlMapping.md   ← AWS/GCP cloud mapping guide for the full curriculum
│
├── result/                 ← PNG + Markdown analysis files saved automatically after each run
│   ├── NumpyStockArray.png
│   ├── NumpyStockArray.md
│   ├── ... (44 files in total)
│
├── requirements.txt        ← required libraries
├── Readme.md               ← this file (English)
└── readme-kr.md            ← Korean version
```

---

## <i class="fa-solid fa-rocket"></i> Quick Start

### Step 1: Requirements

| Item | Recommended version |
|------|---------------------|
| Python | 3.10 or later |
| OS | Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2 recommended) |
| RAM | 4GB or more (8GB recommended for the deep-learning files) |
| Internet | Required to download data with yfinance |

### Step 2: Install

```bash
# 1) Clone the repository
git clone <repo-url>
cd python-ml-class

# 2) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3) Install libraries
pip install --upgrade pip
pip install -r requirements.txt

# 4) Install Korean fonts (Linux/Ubuntu)
sudo apt install fonts-nanum -y
```

### Step 3: First run (about 5 minutes)

```bash
cd src
python NumpyStockArray.py
```

After it finishes, `result/NumpyStockArray.png` is created.

---

## <i class="fa-brands fa-aws"></i> Extending this curriculum on AWS

This repository is currently a **local lab**: it downloads data with yfinance on your PC, then trains and runs models there. When you move it to AWS, do not rewrite the code from scratch. Split the responsibilities of data, training, and deployment as follows.

| Goal | What this repo already does | Recommended AWS resource | When to use it |
|------|-----------------------------|--------------------------|----------------|
| Store raw data and results | yfinance price CSVs, `result/` charts, training artifacts | **Amazon S3** | First |
| Collect prices on a schedule | `YfinanceNormalize.py` | **EventBridge Scheduler + AWS Lambda + S3** | When you need a daily pull |
| Preprocess and build features | Pandas normalization, RSI/MACD, train/validation splits | **SageMaker AI Processing** | When the data grows or you need reproducible batch jobs |
| Train scikit-learn models | Linear/logistic regression, SVM, K-Means, PCA | **SageMaker AI Training** (scikit-learn container) | When you need more CPU than your laptop |
| Train PyTorch models | LSTM, CNN, CNN-LSTM, Transformer | **SageMaker AI Training** (PyTorch container, GPU if needed) | When you run stages 6–7 on a GPU |
| Track experiments and model versions | Parameters, metrics, approved model versions | **SageMaker AI Experiments / Model Registry** | When you compare runs or operate as a team |
| Batch predictions | Next-day or next-week forecasts for many tickers | **SageMaker Batch Transform** | When you do not need a real-time response |
| Deploy the web app | `webapp/backend` Flask API + static UI | **AWS App Runner** or **ECS Fargate** | When you want to serve the current web app as-is |
| Real-time model inference | API that calls a saved PyTorch/scikit-learn model | **SageMaker AI Real-time Endpoint** | After you change the app to save and reuse a trained model |
| Logs and alerts | Training logs, Lambda/App Runner errors | **Amazon CloudWatch** | Every production stage |

> **Recommended starting point:** At classroom scale, connecting `S3 → SageMaker AI Training → S3` is enough. The current `webapp` downloads data and retrains some models on every request, so do not jump straight to a SageMaker Endpoint. First save the trained model to S3 and load it when the app starts.

`Amazon SageMaker AI` is the new name, as of December 2024, for the former Amazon SageMaker. The AWS CLI/API namespace remains `sagemaker` for compatibility. See the [official AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) for the service overview.

### AWS CLI setup

The examples below use the Seoul region (`ap-northeast-2`). Install AWS CLI v2, and prefer an IAM Identity Center (SSO) profile or an IAM role on EC2/CloudShell instead of long-lived access keys.

```bash
# If you use SSO, enter your organization's start URL and region.
aws configure sso --profile ml-class
aws sts get-caller-identity --profile ml-class

# Values shared by the examples below
export AWS_PROFILE=ml-class
export AWS_REGION=ap-northeast-2
export BUCKET="python-ml-class-<unique-bucket-name>"
```

SageMaker jobs need a **SageMaker execution role (Role ARN)** that can read and write S3 and write CloudWatch Logs. Even for a class, do not attach `AdministratorAccess`. Ask an administrator for a role scoped to that S3 bucket. The role's trust principal must be `sagemaker.amazonaws.com`.

```bash
# Look up an existing execution role and store its ARN.
aws iam get-role --role-name SageMakerExecutionRole \
  --query 'Role.Arn' --output text

export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
```

### 1) Upload data and lab results to S3

Use S3 as the smallest unit of a data lake. Bucket names must be globally unique. The commands below are for the Seoul region.

```bash
# Create the bucket and enable default encryption
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$AWS_REGION" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"

aws s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# Example: upload a collected CSV to raw/ and lab results to results/
aws s3 cp ./data/005930.KS.csv "s3://$BUCKET/raw/ticker=005930.KS/"
aws s3 sync ./result "s3://$BUCKET/results/"

# Confirm the upload
aws s3 ls "s3://$BUCKET/" --recursive
```

`./data/005930.KS.csv` is an example path. This repository has no default data folder, so either extend `YfinanceNormalize.py` slightly to save a CSV and then upload it, or have a separate collector Lambda write to the S3 `raw/` prefix.

### (Optional) Schedule weekday close-price collection

If you need a regular pull, package the yfinance collector as a Lambda deployment package or container image, then have an EventBridge rule invoke that Lambda. The time below is **weekdays 20:30 KST (= 11:30 UTC)**. EventBridge cron expressions are in UTC.

```bash
export FUNCTION_NAME=stock-price-collector
export FUNCTION_ARN=$(aws lambda get-function \
  --function-name "$FUNCTION_NAME" \
  --query 'Configuration.FunctionArn' --output text)

aws events put-rule \
  --name daily-stock-pull \
  --schedule-expression 'cron(30 11 ? * MON-FRI *)' \
  --state ENABLED

export RULE_ARN=$(aws events describe-rule --name daily-stock-pull \
  --query 'Arn' --output text)

aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id allow-eventbridge-daily-stock-pull \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN"

aws events put-targets --rule daily-stock-pull \
  --targets "Id"="1","Arn"="$FUNCTION_ARN"
```

Give the Lambda execution role only `s3:PutObject` on the target S3 prefix. If a sensitive value such as an API key appears, store it in AWS Secrets Manager, not in code or on the command line. This example assumes the Lambda function is already deployed.

### 2) Submit SageMaker AI Processing and Training jobs

You can move the normalization and feature engineering in `PandasPortfolio.py` and `YfinanceNormalize.py` to a Processing Job, and the training in `LstmStockPyTorch.py`, `CnnLstmHybrid.py`, and `TransformerAttention.py` to a Training Job. Both jobs read input from S3 and write models and logs to S3 and CloudWatch.

The AWS CLI input schema changes often, so generate a current template from the installed CLI and fill in only the values you need.

```bash
# Processing Job template: fill in the input S3 path, processing image, role, and output S3 path.
aws sagemaker create-processing-job \
  --generate-cli-skeleton input > processing-job.json

# Training Job template: fill in TrainingImage, RoleArn, InputDataConfig,
# OutputDataConfig, ResourceConfig, and StoppingCondition.
aws sagemaker create-training-job \
  --generate-cli-skeleton input > training-job.json

# Submit the jobs from the files you edited
aws sagemaker create-processing-job \
  --cli-input-json file://processing-job.json

aws sagemaker create-training-job \
  --cli-input-json file://training-job.json
```

In `training-job.json`, include this repo's PyTorch training script under `/opt/ml/code/` in the container, read data from `/opt/ml/input/data/train/`, and write the model to `/opt/ml/model/`. Start CPU labs around `ml.m5.xlarge`, and choose a GPU instance such as `ml.g5.xlarge` only if LSTM/CNN/Transformer training is too slow. Always set `StoppingCondition.MaxRuntimeInSeconds` so cost has a ceiling.

```bash
# Check status and the S3 location of the model artifact
aws sagemaker describe-training-job \
  --training-job-name <training-job-name> \
  --query '{Status:TrainingJobStatus,ModelArtifacts:ModelArtifacts.S3ModelArtifacts}'

# Stop a running job you no longer need
aws sagemaker stop-training-job --training-job-name <training-job-name>
```

A SageMaker Training Job stores the model artifact in the S3 path you specify when it finishes. Required fields and cost-limit options are in the [AWS CLI `create-training-job` docs](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-training-job.html). The Processing Job input format is in the [`create-processing-job` docs](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-processing-job.html).

### 3) Deploy a saved model as a real-time API (optional)

If you have changed the web app to reuse a trained model, create a SageMaker AI Endpoint in this order: `create-model → create-endpoint-config → create-endpoint`. An endpoint is billed for its instance even when idle, so for a class lab Batch Transform, or loading the model directly in App Runner, is often cheaper.

```bash
# Generate the latest CLI templates for the model, endpoint config, and endpoint.
aws sagemaker create-model --generate-cli-skeleton input > model.json
aws sagemaker create-endpoint-config --generate-cli-skeleton input > endpoint-config.json
aws sagemaker create-endpoint --generate-cli-skeleton input > endpoint.json

aws sagemaker create-model --cli-input-json file://model.json
aws sagemaker create-endpoint-config --cli-input-json file://endpoint-config.json
aws sagemaker create-endpoint --cli-input-json file://endpoint.json

# Example invoke call when endpoint.json names the endpoint stock-predict-endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name stock-predict-endpoint \
  --content-type application/json \
  --body fileb://request.json response.json
```

When the lab is over, delete resources in reverse order. The endpoint references the endpoint config and the model, so delete the endpoint first.

```bash
aws sagemaker delete-endpoint --endpoint-name stock-predict-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name stock-predict-config
aws sagemaker delete-model --model-name stock-predict-model
```

See the [AWS CLI Endpoint docs](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-endpoint.html) for the exact create, delete, and invoke formats. A broader AWS/GCP mapping and Python SDK examples are in [CloudMlMapping.md](docs/CloudMlMapping.md).
---

## <i class="fa-solid fa-book-open"></i> Learning roadmap

Work through the labs in this order to move systematically from **beginner → intermediate → advanced**.

### <i class="fa-solid fa-circle" style="color:#22c55e;"></i> Stage 1: Python basics and data handling (intro)

| Order | File | Topic | Time |
|------|------|------|------|
| 1 | `korean_font.py` | Korean font utility — understand the import structure | 10 min |
| 2 | `NumpyStockArray.py` | NumPy array math, return and volatility | 20 min |
| 3 | `PandasPortfolio.py` | Pandas DataFrame, missing values, groupby | 20 min |
| 4 | `YfinanceNormalize.py` | Download real prices; Min-Max, Z-score, log return | 25 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** broadcasting, slicing, fillna, and why normalization is needed

---

### <i class="fa-solid fa-circle" style="color:#3b82f6;"></i> Stage 2: Machine learning basics — regression and classification (beginner)

| Order | File | Topic | Time |
|------|------|------|------|
| 5 | `LinearRegressionFundamental.py` | Linear regression: PER/PBR/ROE → expected return | 25 min |
| 6 | `LinearRegressionReturn.py` | Linear regression: past 5-day returns → next 5 days | 30 min |
| 7 | `LogisticTradeSignal.py` | Logistic regression: RSI/MACD → buy/sell classification | 30 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** supervised learning, error functions (MAE/MSE), decision boundary, sigmoid

---

### <i class="fa-solid fa-circle" style="color:#3b82f6;"></i> Stage 3: Deeper machine learning — SVM and unsupervised learning (beginner to intermediate)

| Order | File | Topic | Time |
|------|------|------|------|
| 8 | `SvmTradeSignal.py` | SVM linear kernel: trade-signal classification | 30 min |
| 9 | `SvmMarketPhase.py` | SVM RBF kernel: market regime (up/down) classification | 30 min |
| 10 | `KMeansStockCluster.py` | K-Means: automatic stock clustering | 25 min |
| 11 | `PcaStockReduce.py` | PCA: reduce 6 dimensions to 2 | 25 min |
| 12 | `HyperparamTuning.py` | Grid search, K-fold, overfitting/dropout experiments | 40 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** margin maximization, RBF kernel, centroid, principal components, cross-validation

---

### <i class="fa-solid fa-circle" style="color:#eab308;"></i> Stage 4: Time series and ARIMA (intermediate)

| Order | File | Topic | Time |
|------|------|------|------|
| 13 | `TimeSeriesAnalysis.py` | Moving averages, Bollinger Bands, ACF/PACF, ADF stationarity test | 35 min |
| 14 | `ArimaStockForecast.py` | ARIMA(2,1,2): one-month-ahead price forecast | 35 min |
| 15 | `HeatmapRiskMask.py` | Heatmap and masking: detect high-risk periods | 20 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** stationarity, differencing, AR/MA order, AIC, 95% confidence interval

---

### <i class="fa-solid fa-circle" style="color:#f97316;"></i> Stage 5: Deep learning basics — implement a neural net yourself (intermediate)

| Order | File | Topic | Time |
|------|------|------|------|
| 16 | `NeuralNetBackprop.py` | Backpropagation implemented with NumPy only | 45 min |
| 17 | `RnnBackprop.py` | Vanilla RNN + BPTT implemented from scratch | 45 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** forward pass / backpropagation, chain rule, BPTT, vanishing gradients

---

### <i class="fa-solid fa-circle" style="color:#f97316;"></i> Stage 6: PyTorch deep learning (intermediate to advanced)

| Order | File | Topic | Time |
|------|------|------|------|
| 18 | `LstmStockPyTorch.py` | LSTM: 20 days → next-day price | 45 min |
| 19 | `TimeSeriesWindow.py` | CLI LSTM: choose ticker, window, and horizon | 40 min |
| 20 | `CnnTimeSeriesFeature.py` | 1D CNN: extract patterns from a return sequence | 40 min |
| 21 | `CnnCandleChart.py` | 2D CNN: classify candlestick-chart images | 45 min |
| 22 | `CnnLstmHybrid.py` | CNN + LSTM hybrid | 45 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** cell state, gates, Conv1d/Conv2d, sliding window

---

### <i class="fa-solid fa-circle" style="color:#ef4444;"></i> Stage 7: Transformer (advanced)

| Order | File | Topic | Time |
|------|------|------|------|
| 23 | `TransformerAttention.py` | Multi-head self-attention: direction classification and an attention heatmap | 60 min |

> <i class="fa-solid fa-lightbulb"></i> **Focus:** Q/K/V, positional encoding, visualizing attention weights

---

## How to run each file

Run every file from the `src/` folder.

```bash
cd src
```

### Default run (most files)

```bash
python ArimaStockForecast.py
python CnnCandleChart.py
python HyperparamTuning.py
# ... same command; only the filename changes
```

### Files that take CLI arguments

**TimeSeriesWindow.py** — set the ticker, window, and forecast horizon

```bash
# Defaults: GS P&L (078935.KS), window=20 days, horizon=5 days
python TimeSeriesWindow.py

# Samsung Electronics, 30-day window, 10-day horizon
python TimeSeriesWindow.py --ticker 005930.KS --window 30 --horizon 10

# Apple, a US stock
python TimeSeriesWindow.py --ticker AAPL --window 20 --horizon 5
```

### Check the output

Every file saves a PNG under **`result/`**.

```bash
ls ../result/          # list the generated files
```

---

## <i class="fa-solid fa-book"></i> How to use the learning materials

### `docs/` — explanations written in elementary-school language

Each Python file has a matching Markdown file.

```
docs/NumpyStockArray.md     ← explanation of NumpyStockArray.py
docs/LstmStockPyTorch.md    ← explanation of LstmStockPyTorch.py
...
```

**How to use them:**
1. Read `docs/[filename].md` first to see the overall flow.
2. Open `src/[filename].py` and read the code.
3. Run the code and generate the chart in `result/`.
4. Read `result/[filename].md` to learn how to interpret the result.

### `docs/CloudMlMapping.md` — AWS/GCP cloud mapping guide

This is not a per-file explanation. It walks **across the whole curriculum (23 files + webapp)** and compares the libraries and techniques used at each stage (NumPy/Pandas, scikit-learn, statsmodels, PyTorch, reinforcement learning, and so on) with the matching managed services on AWS and GCP, including real CLI/SDK usage. Use it after the local labs when you want to scale the work to the cloud.

### `result/` — output PNGs and analysis guides

| File pattern | Description |
|----------|------|
| `[filename].png` | Visualization image |
| `[filename]_078935_KS.png` | Result based on real GS P&L prices |
| `[filename].md` | How to read the chart, plus a model-interpretation guide |

---

## <i class="fa-solid fa-lightbulb"></i> Lab tips and stretch exercises

### Stretch exercises after the basic labs

**Stage 1**
- `NumpyStockArray.py`: expand to 10 tickers and compute the Sharpe ratio (mean return / volatility).
- `YfinanceNormalize.py`: switch the ticker to Samsung Electronics (`005930.KS`) and compare.

**Stage 2**
- `LinearRegressionReturn.py`: change the lookback from 5 days to 10 and see whether R² improves.
- `LogisticTradeSignal.py`: raise the threshold from 0.5 to 0.6 and watch how precision changes.

**Stage 3**
- `HyperparamTuning.py`: widen the C range to `[0.001, 0.01, 0.1, 1, 10, 100, 1000]` and find the best value.
- `KMeansStockCluster.py`: try k=4 and k=5 and compare the clusters.

**Stage 4**
- `ArimaStockForecast.py`: compare AIC for `p=1, q=1` and `p=3, q=3`.

**Stages 5–7**
- `LstmStockPyTorch.py`: change `SEQ_LEN=20` to `SEQ_LEN=60` and check the performance change.
- `TimeSeriesWindow.py`: compare several tickers (`AAPL`, `005930.KS`, `078935.KS`).
- `TransformerAttention.py`: increase to `d_model=32, nhead=4` and watch how the attention pattern changes.

### Debugging checklist

```
<i class="fa-solid fa-circle-check"></i> yfinance data error → check the internet connection and ticker (append .KS for Korea Exchange symbols)
<i class="fa-solid fa-circle-check"></i> Broken Korean text → sudo apt install fonts-nanum, then run again
<i class="fa-solid fa-circle-check"></i> CUDA error → force torch.device('cpu')
<i class="fa-solid fa-circle-check"></i> ModuleNotFoundError → run pip install -r requirements.txt again
<i class="fa-solid fa-circle-check"></i> result/ folder missing → confirm you are running from inside src/
```

---

## <i class="fa-solid fa-book"></i> How deep learning (DL) and machine learning (ML) actually work

### 1. Are machine learning and deep learning algorithms all about reducing error?

The short answer is **mostly yes**.

The core goal of machine learning and deep learning is **"to make the computer shrink the gap (the error) between the correct answer and its prediction as much as possible."**

> **<i class="fa-solid fa-bullseye"></i> Simple example: hitting a target**
> The first arrow lands far from the center (large error).
> You look at the miss and correct: "a little more to the right, a little higher."
> Each repetition gets closer to the center.
> Machine learning does the same thing: it corrects itself by the amount it was wrong, and gets more accurate.

The approach still depends on the kind of problem.

---

#### ① Supervised learning: studying with an answer key

* **Examples:** linear regression, logistic regression, image-classification deep learning
* **Idea:** compare the model's prediction with the true answer and reduce the error.
* **Loss function:** the formula that turns that error into a number.
* **Optimization:** the method that drives the error toward zero.

> **<i class="fa-solid fa-book-open"></i> Simple example: a teacher grading a test**
> The student (the model) takes the test → the teacher (the answer key) grades it → the student studies the questions they missed.
> Repeat this and the score (accuracy) goes up.
>
> Stock example: solve "will Samsung Electronics go up or down tomorrow?", compare with the actual result, and keep correcting.

---

#### ② Unsupervised learning: grouping things yourself, with no answer key

* **Examples:** clustering, dimensionality reduction (PCA)
* **Idea:** with no labels, group similar items or find patterns inside the data.

> **<i class="fa-solid fa-apple-whole"></i> Simple example: sorting fruit**
> In a basket of mixed apples, bananas, and grapes, nobody says "this is an apple."
> You still form groups of items that look alike in shape, color, and size.
>
> Stock example: automatically group thousands of stocks into sets that move similarly.

---

#### ③ Reinforcement learning: raising a score, like a game

* **Examples:** AlphaGo, self-driving cars
* **Idea:** every action earns a score (a reward), and the agent learns a strategy that maximizes that score.

> **<i class="fa-solid fa-gamepad"></i> Simple example: a video game**
> The character gets +10 points for catching an enemy and −5 for falling.
> To raise the score, it plays thousands of times and teaches itself how to move.
>
> Stock example: an AI tries "buy" and "sell" countless times and discovers a strategy that makes money.

---

### 2. Is deep learning just linear regression repeated forward and backward?

**That is the right core idea.**

Deep learning is **"stacking a huge number of simple calculations (linear regression), then repeating a forward calculation (forward pass) and a backward correction (backpropagation)."**

---

#### ① Stacking linear regression into something huge (a neural network)

The basic linear-regression formula:

$$\text{Output} = WX + b$$

(multiply the input $X$ by a weight $W$ and add a bias $b$)

The smallest unit of deep learning, the **perceptron**, works the same way. Deep learning is a huge net of thousands or tens of thousands of these units wired together.

> **<i class="fa-solid fa-cube"></i> Simple example: stacking LEGO bricks**
> One brick (linear regression) can only make a simple shape.
> Assemble thousands of them (deep learning) and you can build a robot, a castle, or a car.

> **<i class="fa-solid fa-lightbulb"></i> Nonlinear activation functions**
> If you only stack the bricks, you still only get straight lines.
> Insert a **"bending" function such as ReLU or sigmoid** and you can represent curved shapes.
> That is also how complex real-world price patterns are represented.

---

#### ② Calculating forward: forward propagation

* **Direction:** input layer → hidden layer → output layer (forward)
* **Role:** take the input, produce a final prediction, and compute the error against the answer.

> **<i class="fa-solid fa-book"></i> Simple example: answering a test question**
> Read the question (input) → think (hidden layer) → write the answer (output).
> Grading tells you how many points you missed (the error).

---
#### ③ Correcting backward: backpropagation

* **Direction:** output layer → hidden layer → input layer (backward)
* **Role:** trace the error backward, see how much each unit contributed to the mistake, and nudge the weights.

> **<i class="fa-solid fa-pen"></i> Simple example: an error notebook**
> Look at the missed question again (backward) → trace "where did I slip?" → study that part harder (weight update).
> Repeat this and you solve more of the questions.

---

#### ④ Repeat without end (epoch / iteration)

Repeating the forward pass (solving the problem) and backpropagation (the error notebook) thousands or tens of thousands of times produces a better prediction model.

> **<i class="fa-solid fa-dumbbell"></i> Simple example: sports practice**
> At first you miss most free throws.
> Every day you shoot 100 times (forward pass) and correct how you missed (backpropagation).
> After thousands of repetitions you make almost all of them.

---

### Summary

| Concept | Simple analogy |
|------|----------|
| Machine learning | A student who studies by reviewing every mistake |
| Supervised learning | Studying for a test that has an answer key |
| Unsupervised learning | Grouping similar things yourself, with no answers |
| Reinforcement learning | Practicing a game to raise the score |
| Deep learning | A giant calculator built from tens of thousands of LEGO bricks |
| Forward pass | Answering the test |
| Backpropagation | Writing the error notebook |
| Repeated training (epoch) | Practicing a little every day |

Deep learning is **a giant assembly of linear regressions with nonlinearity added**, and it gets more accurate by repeating the **pendulum motion of a forward pass (solving the problem) and backpropagation (the error notebook)**.

---

## <i class="fa-solid fa-robot"></i> What is an LLM (large language model)? — what "large" means

> <i class="fa-solid fa-circle-info"></i> Note: this repo has **no rule such as "you must output this many words or sentences to be an LLM."** The real dividing line is not how long the model's sentences are. It is the three "weight class" measures below: **parameter count, training-data volume, and context window**.

### 1. Parameter count — the most common measure

| Class | Parameter count | Examples |
|------|--------------|------|
| Small | Under 1 billion (1B) | This repo's `TransformerAttention.py` model (7,202) |
| Medium | 1 billion to 10 billion | Google Gemma, Meta Llama 3 8B |
| Large (LLM) | 10 billion (10B) and up, into the hundreds of billions | GPT-3 (175 billion) |

> <i class="fa-solid fa-lightbulb"></i> **Recent trend — sLLMs (SLMs)**
> **sLLMs (small large language models)**, cut down to about 1–7 billion parameters so they can run lightly on a phone or laptop, are popular too. The weight class is smaller, but they can be nearly as capable as an LLM.

### 2. Amount of training data — hundreds of billions to trillions of tokens

Modern LLMs usually train on **trillions of tokens**. The name "large language model" applies when the model has absorbed books, news, papers, Wikipedia, and text from websites around the world.

### 3. Context window — the size of the room it can remember at once

What matters more than "how many words can it say?" is **"how many words can it remember and understand at once?"**

- Early models: forget the beginning after about one page (a few thousand words)
- Current LLMs: remember dozens of books at once, often hundreds of thousands to a million tokens

> <i class="fa-solid fa-bullseye"></i> **One-line summary**
> The "Large" in LLM is not about long sentences. The name comes from a **parameter size** and a **training-data volume** that are almost hard to imagine.

---

### <i class="fa-solid fa-radio"></i> What exactly is a parameter?

If a deep-learning model is a huge "route-finding map," parameters are the **numbers written on the road signs (weights W and biases b)**. Billions of parameters means the AI's brain has billions of tiny adjustable screws (connections).

> **<i class="fa-solid fa-sliders"></i> Simple example: knobs on an audio mixer**
> A radio with 3 knobs (volume / tone / bass) is easy to adjust, but it cannot make a delicate sound. A mixer with 70 billion knobs, tuned very finely, can imitate every sound in the world (human language, knowledge, and context). A parameter is one of those knobs.

| Parameter scale | What it can do |
|---------------|-----------|
| Tens of thousands | Simple addition, telling dogs from cats in photos |
| Billions to hundreds of billions | An LLM that can write poetry, fiction, and code, and hold a philosophical conversation |

### <i class="fa-solid fa-code"></i> Python example in this repo: `TransformerAttention.py`

This repo does not contain a real LLM such as GPT, but [TransformerAttention.py](src/TransformerAttention.py) implements and trains the core LLM structure, **multi-head self-attention**, at the smallest scale. It is also the only lab where you can see the abstract idea of a "parameter" in actual code.

```python
# src/TransformerAttention.py (around lines 140–142)
model = StockTransformer(d_model=D_MODEL, nhead=4, num_layers=2, seq_len=SEQ_LEN)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
```

`model.parameters()` walks every weight W and bias b tensor in the model, and `p.numel()` counts the numbers in each tensor (the parameters) and adds them up. Running this repo's model prints:

| Item | This repo's `StockTransformer` | GPT-3 |
|------|------------------------------|-------|
| Embedding dimension (`d_model`) | 16 | 12,288 |
| Attention heads (`nhead`) | 4 | 96 |
| Layer count (`num_layers`) | 2 | 96 |
| **Total parameters** | **7,202** | **175 billion (about 24 million times larger)** |

> <i class="fa-solid fa-flask"></i> **Check it yourself**
> In `StockTransformer.__init__` at [line 136](src/TransformerAttention.py#L116) of `TransformerAttention.py`, increase `d_model`, `nhead`, `num_layers`, and `dim_ff`, then run again. The printed total-parameter count grows as you increase those values. (This is the same exercise as "`d_model=32, nhead=4`" in the stretch-exercise section.)

---

## <i class="fa-solid fa-droplet"></i> What is a diffusion model? — making pictures from noise

> <i class="fa-solid fa-circle-info"></i> **Is this used in this repo? No.** This repo focuses on price prediction, classification, and anomaly detection, so an image-generation diffusion model (forward noise diffusion plus reverse denoising) is not implemented. The notes below are only for the concept, and they compare the closest cousin file and how it differs.

Diffusion here is the same "diffusion" you learn in science class. A drop of ink in water spreads until the whole glass is cloudy. Perfume sprayed in a room fills the room with scent. **A diffusion model is an AI that uses that diffusion process backward, mathematically, to create images.**

### <i class="fa-solid fa-arrows-left-right"></i> How the two stages work

| Stage | Name | What it does | Analogy |
|------|------|---------|------|
| 1 | Forward process | Mix a tiny amount of noise into a sharp photo, hundreds of times, until it is pure noise (TV static) | Dropping ink into clear water one drop at a time until the whole glass is cloudy |
| 2 | Reverse process | From pure noise, repeat "undo just the dust that just diffused" hundreds of times to restore the original photo | Magically gathering the ink particles back into the original drop |

> **<i class="fa-solid fa-wand-magic-sparkles"></i> When you type "draw a retriever in a spacesuit" into Midjourney or Stable Diffusion**
> 1. The AI starts from one completely random noise image (a static screen).
> 2. With the reverse-diffusion skill it learned, it peels the noise away a little at a time and sculpts the shape the prompt asked for.

> <i class="fa-solid fa-bullseye"></i> **One-line summary**
> The name **diffusion model** comes from implementing the scientific idea of diffusion (order → disorder) in math, then reversing it (disorder → order) to make a picture out of nothing but noise.

### <i class="fa-solid fa-code-compare"></i> The closest (but different) example in this repo: `AutoencoderAnomalyDetect.py`

[AutoencoderAnomalyDetect.py](src/AutoencoderAnomalyDetect.py) only *feels* like diffusion because it "damages an input and then restores it." The mechanism is completely different.

| Comparison | Diffusion model | `AutoencoderAnomalyDetect.py` |
|-----------|----------------|-------------------------------|
| How the input is damaged | Noise is added gradually over hundreds of steps (forward process) | No damage step — the input is compressed straight to a low dimension (8) |
| How it is restored | From the noisy state, step back one step at a time (reverse process, hundreds of repeats) | The compressed latent vector is restored by the decoder **in a single pass** |
| Purpose | **Generate** new data (images) from noise | **Detect anomalies** by seeing whether a normal pattern reconstructs well (generation is not the goal) |
| Where the code is | (not in this repo) | [lines 104–119](src/AutoencoderAnomalyDetect.py#L104-L119) `StockAutoencoder` — `encoder: 30→16→8`, `decoder: 8→16→30` |

> <i class="fa-solid fa-lightbulb"></i> If you look for a connection, both share the idea of "knocking information down once and standing it back up." An autoencoder is **single-step compression and reconstruction**. Diffusion is **gradual noise added and removed over hundreds of steps**. They are fundamentally different algorithms. In this repo, the feeling of "repeated refinement over many steps" is closer to the repeated differencing in [ArimaStockForecast.py](src/ArimaStockForecast.py), or to the epoch loop in each training file — but those are still not a real diffusion noise schedule.

---

## File-by-file summary

### 1) Basic data handling

- [NumpyStockArray.py](src/NumpyStockArray.py): price arrays, return/volatility, expected portfolio return
  > Example: compute "how did 5 stocks rise over 8 days?" as a numeric array.

- [PandasPortfolio.py](src/PandasPortfolio.py): fill missing portfolio values, compute market value and return, aggregate by sector
  > Example: fill gaps in a trade log and summarize your return in a table.

### 2) Regression (supervised learning)

- [LinearRegressionReturn.py](src/LinearRegressionReturn.py): predict the next 5-day cumulative return from the previous 5 daily returns
  > Example: draw a line that answers "if the last 5 days moved like this, how much will the next 5 rise?"

- [LinearRegressionFundamental.py](src/LinearRegressionFundamental.py): predict expected return from PER/PBR/ROE
  > Example: read a company's report card (PER, PBR, ROE) and predict "how much will this stock rise?"

- [LogisticTradeSignal.py](src/LogisticTradeSignal.py): classify buy/sell signal probability from RSI/MACD *(logistic regression)*
  > Example: output a trade signal as a probability, such as "RSI 65, positive MACD → 78% chance of buy."

### 3) Classification models (supervised learning)

- [SvmTradeSignal.py](src/SvmTradeSignal.py): classify buy/sell signals from momentum and volume-change rate *(SVM)*
  > Example: look at two indicators and split "buy now or sell now" with a single line.

- [SvmMarketPhase.py](src/SvmMarketPhase.py): classify market regime (up/down) from RSI/MACD/volatility *(SVM RBF)*
  > Example: use three indicators to decide "is the market rising or falling right now?"

### 4) Unsupervised learning

- [KMeansStockCluster.py](src/KMeansStockCluster.py): automatically cluster stock types from return and volatility *(K-Means)*
  > Example: classify 60 stocks into growth, value, and defensive groups with nobody teaching the labels.

- [PcaStockReduce.py](src/PcaStockReduce.py): compress 6 stock indicators into 2 dimensions and plot them *(PCA)*
  > Example: reduce PER, PBR, ROE, volatility, and other numbers from 6 down to 2 so they fit on a chart.

### 5) Helper utility

- [korean_font.py](src/korean_font.py): utility that picks a Korean font for the current OS
  > Example: automatically set a font so Korean titles and axis labels do not break in matplotlib charts.

### 6) Time-series analysis

- [TimeSeriesAnalysis.py](src/TimeSeriesAnalysis.py): moving averages, Bollinger Bands, ACF/PACF, ADF stationarity test
  > Example: analyze the price path with several methods and mathematically check "is this data regular?"

- [ArimaStockForecast.py](src/ArimaStockForecast.py): forecast future prices with ARIMA *(statistical time series)*
  > Example: write the past price pattern and its errors as a formula and forecast the price one month ahead.

### 7) RNN and LSTM (recurrent networks)

- [RnnBackprop.py](src/RnnBackprop.py): vanilla RNN implemented in NumPy *(BPTT)*
  > Example: answer "will today's price rise?" while remembering the last 10 days of returns.

- [LstmStockPyTorch.py](src/LstmStockPyTorch.py): price time-series forecast with a PyTorch LSTM
  > Example: separate long-term memory (cell state) from short-term memory (hidden state) to forecast the price path more accurately.

### 8) CNN (convolutional networks)

- [CnnTimeSeriesFeature.py](src/CnnTimeSeriesFeature.py): 1D CNN extracts patterns from a 30-day return sequence and classifies up/down
  > Example: filters automatically find local features such as three rising days in a row.

- [CnnCandleChart.py](src/CnnCandleChart.py): render a candlestick chart as a 32×32 color image with matplotlib, then classify it with a 2D CNN
  > Example: photograph the candlestick chart and recognize patterns such as consecutive up candles or a down-candle reversal.

- [CnnLstmHybrid.py](src/CnnLstmHybrid.py): CNN compresses short-window patterns into a vector, then LSTM learns the time order *(hybrid)*
  > Example: CNN = a one-day news summary, LSTM = the weekly trend. The two strengths are combined.

### 9) Transformer and attention

- [TransformerAttention.py](src/TransformerAttention.py): multi-head self-attention learns relationships between dates and classifies direction
  > Example: like underlining important words while reading, the model learns which day's price matters for the current prediction.

### 10) Hyperparameter tuning and validation

- [HyperparamTuning.py](src/HyperparamTuning.py): K-fold cross-validation, grid search, overfitting experiments (C and dropout)
  > Example: cook every recipe and keep the tastiest one — an exhaustive search of parameter combinations.

### 11) Data normalization and preprocessing

- [YfinanceNormalize.py](src/YfinanceNormalize.py): download real prices with yfinance, then compare Min-Max, Z-score, and log return
  > Example: to plot Samsung (70,000 won) and AAPL (150 dollars) on the same chart, you have to "put them on the same scale."

### 12) Time-series window forecaster (CLI)

- [TimeSeriesWindow.py](src/TimeSeriesWindow.py): take ticker, window, horizon, and step, then forecast with an LSTM *(interactive/CLI)*
  > Defaults: GS P&L (078935.KS), 20-day window, 5-day horizon, step=1
  > CLI: `python TimeSeriesWindow.py --ticker AAPL --window 30 --horizon 5 --step 1`

### 13) Deep learning and visualization

- [NeuralNetBackprop.py](src/NeuralNetBackprop.py): time-series price forecast with backpropagation implemented by hand *(neural network)*
  > Example: implement the error notebook (backpropagation) yourself and predict tomorrow from the last 5 days of prices.

- [HeatmapRiskMask.py](src/HeatmapRiskMask.py): mask high-risk cells on a return heatmap
  > Example: draw returns as a color map and highlight the risky cells.

---

## <i class="fa-solid fa-chart-bar"></i> Result image guide (`result` folder)

Running each file saves an image under `result/`.

---

### SvmTradeSignal_078935_KS.png — SVM trade-signal boundary

![SvmTradeSignal](result/SvmTradeSignal_078935_KS.png)

**What does it show?**
The chart is split into a **pink region** and a **blue region**, with a diagonal boundary in the middle.

- **Red points** = buy signal (a time to buy)
- **Blue points** = sell/wait signal (a time to sell or wait)
- **Diagonal boundary** = the line SVM found for "should I buy or not?"

> Higher momentum (to the right) and a higher volume-change rate (upward) produce a buy signal.

---

### SvmMarketPhase_078935_KS.png — market-regime prediction from RSI/MACD

![SvmMarketPhase](result/SvmMarketPhase_078935_KS.png)

**What does it show?**
The x-axis is RSI (overbought/oversold) and the y-axis is MACD (trend strength).

- **Red points** = days predicted as a rising market
- **Blue points** = days predicted as a falling market

> Red points cluster when RSI is high (60 or above) and MACD is positive.

---

### NeuralNetBackprop_078935_KS.png — neural-net backpropagation price forecast

![NeuralNetBackprop](result/NeuralNetBackprop_078935_KS.png)

**What does it show?**
It compares the blue solid line (actual price) with the orange dashed line (predicted price).

> The network captures the **overall direction (the trend)** well, but misses **short, sharp swings**.

---

### HeatmapRiskMask.png — return heatmap and high-risk masking

![HeatmapRiskMask](result/HeatmapRiskMask.png)

| Panel | Contents |
|------|------|
| **Left (original)** | Original return map: green = gain, red = loss |
| **Middle (mask)** | Only cells whose return exceeds ±2% are shown in white |
| **Right (highlight)** | Those risky cells are emphasized in a darker color |

---

### KMeansStockCluster.png — automatic K-Means stock-type classification

![KMeansStockCluster](result/KMeansStockCluster.png)

- **Red points (growth)** = high return and high volatility
- **Blue points (value)** = medium return and medium volatility
- **Green points (defensive)** = low return but also low volatility
- **Black X** = each group's center (centroid)

---

### PcaStockReduce.png — PCA dimensionality-reduction result

![PcaStockReduce](result/PcaStockReduce.png)

- Left: what percent of the total information each principal component holds
- Right: scatter plot of 6 indicators compressed to 2 dimensions (color = return)

---

### TimeSeriesAnalysis_078935_KS.png — time-series analysis overview

![TimeSeriesAnalysis](result/TimeSeriesAnalysis_078935_KS.png)

- **Row 1:** moving averages (SMA5/20/60) and Bollinger Bands
- **Row 2:** daily returns (the differenced series)
- **Row 3:** ACF and PACF (hints for choosing the ARIMA order)

---

### ArimaStockForecast_078935_KS.png — ARIMA price forecast

![ArimaStockForecast](result/ArimaStockForecast_078935_KS.png)

- **Top panel:** full price history + 22-trading-day forecast + 95% confidence interval (pink band)
- **Bottom panel:** last 6 months zoomed in, with forecast markers

---

### RnnBackprop_078935_KS.png — vanilla RNN training result

![RnnBackprop](result/RnnBackprop_078935_KS.png)

- **Left:** BCE loss curve (starts near 0.693 → vanishing-gradient limit)
- **Right:** predicted-direction bars (gold = correct, black = wrong)

---

### LstmStockPyTorch_078935_KS.png — PyTorch LSTM price forecast

![LstmStockPyTorch](result/LstmStockPyTorch_078935_KS.png)

- **Left:** MSE loss curve (drops sharply, then converges)
- **Right:** actual (blue) vs LSTM prediction (red)

---

### LogisticTradeSignal_078935_KS.png — logistic-regression buy/sell

![LogisticTradeSignal](result/LogisticTradeSignal_078935_KS.png)

- **Green background:** region where buy probability is high
- **Red background:** region where sell/wait probability is high

---

### CnnTimeSeriesFeature_078935_KS.png — 1D CNN time-series features

![CnnTimeSeriesFeature](result/CnnTimeSeriesFeature_078935_KS.png)

- Training loss and accuracy, predicted-probability bars, and a 4-panel Conv1d feature map

---

### CnnCandleChart_078935_KS.png — 2D CNN candlestick-image classification

![CnnCandleChart](result/CnnCandleChart_078935_KS.png)

- Six sample candlestick images, plus training loss and accuracy

---

### CnnLstmHybrid_078935_KS.png — CNN+LSTM hybrid classification

![CnnLstmHybrid](result/CnnLstmHybrid_078935_KS.png)

- Structure: CNN (short-term pattern) → LSTM (time flow) → Linear (classification)

---

### TransformerAttention_078935_KS.png — Transformer self-attention

![TransformerAttention](result/TransformerAttention_078935_KS.png)

- **Self-attention heatmap:** which dates the model looks at when it predicts

---

### HyperparamTuning.png — hyperparameter tuning and validation

![HyperparamTuning](result/HyperparamTuning.png)

- Grid-search heatmap, overfitting curve for C, dropout effect, and K-fold results

---

### YfinanceNormalize_078935_KS.png — yfinance normalization comparison

![YfinanceNormalize](result/YfinanceNormalize_078935_KS.png)

- Five panels: raw, Min-Max, Z-score, log return, and the distribution

---

### TimeSeriesWindow_078935_KS_w20_h5.png — time-series window forecast

![TimeSeriesWindow](result/TimeSeriesWindow_078935_KS_w20_h5.png)

- Four panels: window samples, LSTM loss, prediction vs actual, and a summary of the run settings

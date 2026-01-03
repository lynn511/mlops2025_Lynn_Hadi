# mlops2025_Lynn_Hadi

# Section1: Pipeline Execution (Makefile)


This project uses a Makefile to orchestrate the ML pipeline.

### Available Commands 
-make preprocess

-make features

---
# Project Structure

```
mlops2025_Lynn_Hadi/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py

├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── batch_inference.py

├── src/
│   ├── mlproject/
│   │   ├── __init__.py
│   │   ├── data/
│   │   ├── preprocess/
│   │   ├── features/
│   │   ├── train/
│   │   ├── inference/
│   │   ├── pipelines/
│   │   └── utils/
│   └── mlproject.egg-info/

├── notebooks/        # exploration only (gitignored)
│   └── EDA.ipynb

├── configs/
└── tests/
```
---
# Section 2 

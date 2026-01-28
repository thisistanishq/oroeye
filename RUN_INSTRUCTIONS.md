# How to Run OroLight

The project is fully set up with a virtual environment and a trained model.

## Prerequisites
- Python 3.13 (installed)
- Virtual Environment (created in `oral_cancer_app/venv`)

## Quick Start

1. **Open a terminal** and navigate to the project folder:
   ```bash
   cd /Users/tanishq/Downloads/orolight-main/oral_cancer_app
   ```

2. **Run the Application** (using the pre-configured virtual environment):
   ```bash
   ./venv/bin/python app.py
   ```

3. **Access the App**:
   - Open your browser to: [http://127.0.0.1:5001](http://127.0.0.1:5001)

## (Optional) Retrain the Model
If you want to retrain the model with new data:
```bash
./venv/bin/python train_model.py
```
This will overwrite `model/best_model.keras`.

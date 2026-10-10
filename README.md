# Telco Customer Churn Predictor

A Streamlit app that predicts whether a telecom customer is likely to churn, using a Random Forest classifier trained on the IBM Telco Customer Churn dataset.
live demo link :  https://telco-churn-prediction-p.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` needs `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder and creates `churn_model.joblib`.

## Model comparison

All models were trained on the same 80/20 stratified split (`random_state=42`) with scaled features. Precision, recall and F1 are for the churn class.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Random Forest (balanced)** | 0.753 | 0.523 | 0.797 | 0.631 | 0.843 |
| Hist Gradient Boosting (balanced) | 0.744 | 0.511 | 0.805 | 0.625 | 0.843 |
| Bagging of Decision Trees (balanced) | 0.747 | 0.515 | 0.791 | 0.624 | 0.841 |
| Decision Tree (balanced) | 0.735 | 0.500 | 0.807 | 0.618 | 0.831 |
| SVC (balanced) | 0.747 | 0.516 | 0.767 | 0.617 | 0.819 |
| Logistic Regression (balanced) | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| AdaBoost | 0.805 | 0.662 | 0.540 | 0.595 | 0.841 |
| Gradient Boosting | 0.798 | 0.654 | 0.505 | 0.570 | 0.842 |
| KNN | 0.747 | 0.525 | 0.500 | 0.512 | 0.772 |

## Observations

- **Random Forest** has the best F1 and the best accuracy among the balanced models, with recall close to the top. It was chosen as the final model.
- **Hist Gradient Boosting** is almost identical to Random Forest, with slightly higher recall and slightly lower accuracy and F1.
- **Bagging of Decision Trees** is very close to Random Forest but slightly lower on every metric.
- **Decision Tree** has the highest recall of all, but lower precision and AUC. A single tree is less stable than an ensemble.
- **SVC** has the lowest recall among the balanced models and the lowest AUC among the strong models.
- **Logistic Regression (balanced)** performs about as well as the tree ensembles, showing the data is largely linearly separable.
- **Logistic Regression (unbalanced)** has the highest accuracy and precision, but it misses about 43% of churners (recall 0.567).
- **AdaBoost** and **Gradient Boosting** show the same pattern: high accuracy and precision but low recall, because they were trained without class weights. This makes them weaker at finding churners.
- **KNN** is the weakest on F1, recall and AUC. It does not handle the imbalanced classes and the many one-hot features well.
- Models trained with balanced class weights trade accuracy for recall. For churn prediction, finding the customers who will leave matters more than overall accuracy, so those models were preferred.
- Differences between the top models are small (about 1 point of F1), so 5-fold cross-validation was also used. Random Forest stayed on top with F1 0.635, accuracy 0.756 and AUC 0.846.

## Files

- `app.py` - Streamlit app
- `train_model.py` - trains the model and saves it
- `requirements.txt` - dependencies

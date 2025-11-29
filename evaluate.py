from sklearn.metrics import classification_report

def evaluate_best_model(model, df):
    X = df.drop('booking_status', axis=1)
    y = df['booking_status'].apply(lambda x: 1 if x=='Canceled' else 0)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

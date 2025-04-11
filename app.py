from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the full pipeline
model = joblib.load("best_churn_model.pkl")

# Generate simple human-readable explanation
def generate_explanation(data, prediction_label):
    contributing = []

    if int(data['Recency']) > 300:
        contributing.append("High Recency")

    if int(data['Frequency']) <= 2:
        contributing.append("Low Frequency")

    if float(data['Total_Amount']) < 100:
        contributing.append("Low Total Amount")

    if float(data['Return_Rate']) > 0.5:
        contributing.append("High Return Rate")

    if not contributing:
        return "No strong contributing factors detected."

    return "Contributing Factors: " + ", ".join(contributing)


# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction form page
@app.route('/prediction')
def prediction_form():
    return render_template('prediction.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        form_data = {
            "Gender": request.form['gender'],
            "Age": request.form['age'],
            "Recency": request.form['recency'],
            "Frequency": request.form['frequency'],
            "Total_Amount": request.form['total_amount'],
            "Unique_Categories": request.form['unique_categories'],
            "Avg_Purchase_Value": request.form['avg_purchase_value'],
            "Return_Rate": request.form['return_rate']
        }

        # Convert values to appropriate types
        input_data = {
            "Gender": form_data["Gender"],
            "Age": int(form_data["Age"]),
            "Recency": int(form_data["Recency"]),
            "Frequency": int(form_data["Frequency"]),
            "Total_Amount": float(form_data["Total_Amount"]),
            "Unique_Categories": int(form_data["Unique_Categories"]),
            "Avg_Purchase_Value": float(form_data["Avg_Purchase_Value"]),
            "Return_Rate": float(form_data["Return_Rate"])
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        prediction_label = "Yes" if prediction == 1 else "No"
        prediction_text = f"Churn Prediction: {prediction_label} (Probability: {proba:.2%})"
        explanation = generate_explanation(form_data, prediction_label)

        return render_template("prediction.html",
                               prediction_text=prediction_text,
                               explanation=explanation,
                               form_data=form_data)

    except Exception as e:
        return render_template("prediction.html",
                               prediction_text=f"Error: {str(e)}",
                               form_data=form_data)

@app.route('/customer', methods=['GET', 'POST'])
def customer_analysis():
    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        
        try:
            # Load the dataset
            df = pd.read_csv(r"C:\Users\tjtha\Videos\Bandicam\Sem4\churn_app\cleaned_ecommerce_data.csv")
            
            # Calculate Total Purchase if it doesn't exist
            if 'Total Purchase' not in df.columns:
                df['Total Purchase'] = df['Product Price'] * df['Quantity']
            
            # Convert Customer ID to string if needed
            df['Customer ID'] = df['Customer ID'].astype(str)
            
            # Filter data for the specific customer
            customer_data = df[df['Customer ID'] == customer_id.strip()]
            
            if customer_data.empty:
                return render_template('customer.html', error=f"No data found for Customer ID: {customer_id}")
            
            # Calculate metrics
            metrics = {
                'total_spent': customer_data['Total Purchase'].sum(),
                'avg_purchase': customer_data['Total Purchase'].mean(),
                'total_orders': len(customer_data),
                'favorite_category': customer_data['Product Category'].mode()[0],
                'last_purchase': customer_data['Purchase Date'].max(),
                'return_rate': (customer_data['Returns'].sum() / len(customer_data)) * 100
            }
            
            # Prepare chart data
            monthly_spending = customer_data.groupby('Purchase Month')['Total Purchase'].sum()
            category_dist = customer_data['Product Category'].value_counts()
            
            return render_template('customer.html',
                                customer_id=customer_id,
                                metrics=metrics,
                                monthly_spending=monthly_spending.to_dict(),
                                category_dist=category_dist.to_dict(),
                                purchases=customer_data.to_dict('records'))
            
        except Exception as e:
            return render_template('customer.html', error=str(e))
    
    return render_template('customer.html')

if __name__ == '__main__':
    app.run(debug=True)
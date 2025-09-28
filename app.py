import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

SAVED_MODEL_DIR = "./saved_models"

# Load the pre-trained models and scalers
scalar = joblib.load(os.path.join(SAVED_MODEL_DIR, "scaler.pkl"))
linear_model = joblib.load(os.path.join(SAVED_MODEL_DIR, "linear_model.pkl"))
poly_model = joblib.load(os.path.join(SAVED_MODEL_DIR, "poly_model.pkl"))
logistic_model = joblib.load(os.path.join(SAVED_MODEL_DIR, "logistic_model.pkl"))
rf_classifier = joblib.load(os.path.join(SAVED_MODEL_DIR, "rf_classifier.pkl"))


# AQI category function
def get_aqi_category(aqi):
    if aqi <= 200:
        return "Good 😊", "Air quality is safe. Enjoy outdoor activities!", "green"
    elif aqi <= 150:
        return "Moderate 😐", "Acceptable air quality, but sensitive groups should take care.", "yellow"
    elif aqi <= 100:
        return "Unhealthy for Sensitive Groups 🤧", "Sensitive groups should reduce outdoor exertion.", "orange"
    elif aqi <= 50:
        return "Unhealthy 😷", "Everyone may begin to feel adverse health effects.", "red"
    elif aqi <= 10:
        return "Very Unhealthy 🛑", "Health alert: Avoid outdoor activities.", "purple"
    else:
        return "Hazardous ☠️", "Health warning: Stay indoors, use purifier.", "maroon"


def predict_aqi(pm25, pm10, no2, co, temp, humidity):
    # Prepare input
    input_data = pd.DataFrame(
        [[pm25, pm10, no2, co, temp, humidity]],
        columns=["PM2.5", "PM10", "NO2", "CO", "Temperature", "Humidity"]
    )
    input_scaled = scalar.transform(input_data)

    # Predictions
    linear_pred = linear_model.predict(input_scaled)[0]
    poly_pred = poly_model.predict(input_scaled)[0]
    logistic_class = logistic_model.predict(input_scaled)[0]
    rf_class = rf_classifier.predict(input_scaled)[0]

    # Plot
    models = ["Linear Regression", "Polynomial Regression"]
    predictions = [linear_pred, poly_pred]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=predictions, palette="coolwarm")
    plt.title("AQI Predictions by Regression Models")
    plt.ylabel("Predicted AQI")
    plt.ylim(0, max(predictions) + 50)
    plt.tight_layout()
    plt.savefig("aqi_plot.png")
    plt.close()

    # AQI category info (based on Linear model)
    category, advisory, color = get_aqi_category(linear_pred)

    # Classification results with emoji
    logistic_status = "✅ Safe" if logistic_class == 0 else "❌ Unsafe"
    rf_status = "✅ Safe" if rf_class == 0 else "❌ Unsafe"

    # Final Markdown output (with AQI card)
    output_text = f"""
        <div style="border-radius:12px; padding:15px; background-color:{color}; color:white; font-size:18px;">
            <b>🌍 Air Quality Prediction:</b><br>
            <b>Linear Regression AQI:{linear_pred:.2f}</b><br>
            <b>Polynomial Regression AQI:{poly_pred:.2f}</b><br>
            <b>🧪 Classification Results:</b><br>
            <b>Logistic Model :{logistic_status}</b><br>
            <b>Random Forest :{rf_status}</b><br>
            <b>Category:</b> {category}<br>
            <b>Advice:</b> {advisory}<br>
            
            
        </div>
        
    """
    return output_text, "aqi_plot.png"


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_aqi,
        inputs=[
            gr.Slider(0, 200, label="PM2.5 (µg/m³)", value=50),
            gr.Slider(0, 300, label="PM10 (µg/m³)", value=80),
            gr.Slider(0, 100, label="NO2 (µg/m³)", value=20),
            gr.Slider(0, 10, label="CO (mg/m³)", value=1),
            gr.Slider(-10, 40, label="Temperature (°C)", value=20),
            gr.Slider(0, 100, label="Humidity (%)", value=50)
        ],
        outputs=[
            gr.Markdown(label="📊 Prediction Report"),
            gr.Image(label="Model Comparison Plot")
        ],
        title="🌫️ Air Quality Prediction and Classification",
        description="Enter pollutant levels and weather conditions to predict AQI and classify air quality. Built with multiple machine learning models."
    )

    iface.launch()
# Bangladesh Flight Price Predictor

A web application that predicts flight prices in Bangladesh using machine learning. This project uses a Random Forest Regressor model trained on historical flight data to provide accurate price predictions.

![Project Screenshot](screenshot.png)

## Features

- 🎯 Accurate flight price predictions
- 🌐 User-friendly web interface
- 📱 Responsive design for all devices
- ⚡ Real-time predictions
- 📊 Multiple input parameters for precise predictions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
  https://drive.google.com/file/d/1e_9Zh-zi8wB6O8QKa14XsB7z410iC6Hu/view?usp=drive_link

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Flight-Price-Prediction-Using-Machine-Learning-Approach.git
cd Flight-Price-Prediction-Using-Machine-Learning-Approach
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Flight-Price-Prediction-Using-Machine-Learning-Approach/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface template
├── best_flight_price_model.tkl    # Trained model
└── flight_price_scaler.tkl        # Feature scaler
```
![Screenshot 2025-05-04 090254](https://github.com/user-attachments/assets/9a47453a-3d43-4d26-91dc-6b4bce718faa)

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Fill in the flight details:
   - Select airline
   - Choose source and destination
   - Specify aircraft type
   - Select class
   - Choose booking source
   - Select seasonality
   - Specify stopovers
   - Enter duration (hours)
   - Enter days before departure

4. Click "Predict Price" to get the estimated flight price

## Model Information

The prediction model uses a Random Forest Regressor trained on historical flight data. It considers the following factors:

- Airline
- Source and destination cities
- Aircraft type
- Travel class
- Booking source
- Seasonality
- Number of stopovers
- Flight duration
- Days before departure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Flight Price Dataset of Bangladesh](https://www.kaggle.com/datasets/mahatiratusher/flight-price-dataset-of-bangladesh)
- Machine Learning Model: Random Forest Regressor
- Web Framework: Flask
- Frontend: Bootstrap 5

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/Flight-Price-Prediction-Using-Machine-Learning-Approach](https://github.com/yourusername/Flight-Price-Prediction-Using-Machine-Learning-Approach)

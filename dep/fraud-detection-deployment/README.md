# Fraud Detection Deployment

This project is designed to deploy a fraud detection model using Docker. The model is trained to identify fraudulent transactions and is served through a web application.

## Project Structure

```
fraud-detection-deployment
├── app
│   ├── main.py
│   ├── model
│   │   └── Transaction_fraud_detection.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Docker installed on your machine.
- Basic knowledge of Docker and Python.

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fraud-detection-deployment
   ```

2. Build the Docker image:
   ```
   docker build -t fraud-detection-app .
   ```

### Running the Application

1. Run the Docker container:
   ```
   docker run -p 5000:5000 fraud-detection-app
   ```

2. Access the application:
   Open your web browser and go to `http://localhost:5000`.

### API Endpoints

- **POST /predict**: Use this endpoint to send transaction data for fraud detection. The request should contain the necessary features for the model to make a prediction.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Thanks to the contributors and libraries that made this project possible.
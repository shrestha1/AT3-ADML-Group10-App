# AT3-ADML-Group10-App
Machine Learning As a Service

This project comprises the frontend and backend services of a Air Fare prediction application. The app folder contains all components related to the frontend, while the fast-api-backend folder houses the backend services. The backend is responsible for providing API endpoints for air fare prpediction utilizing machine learning models stored in pickle files.

## Technology Stack Used
- For Frontend development
    - Streamlit 
- For Backend development
    - FastAPI
- For containerization:
    - Docker and Docker Compose
- For Deployment:
    - Render


## Project Structure:
```
AT3-ADML-Group10-App/
├── app/                     <- Frontend application
│   ├── config.py            <- Configuration file to configure endpoint (local or remote)
│   ├── main.py              <- Main application running file
│   ├── requirements.txt     <- Packages required to build frontend 
│   └── Dockerfile           <- Dockerfile for the frontend
│
├── fastapi-backend/         <- FastAPI backend
│   ├── models/              <- Directory for model files
│   │   ├── dipesh/          <- Predictive models
│   │   │   ├── cat_encoder.pkl       <- custom Categorical Encoder model
│   │   │   └── shrestha_dipesh_light_GBM.pkl <- Predictive model file
│   │   └── xx/         <- 
│   │       └── xx.pkl  <- prediction model file
│   │
│   ├── src/                 <- Source code for backend processing
│   │   └── features/         <- Feature-related data manipulation
│   │       └── encoder.py <- For processing and transforming data 
│   │
│   ├── routes.py             <- Main FastAPI application for all endpoints
│   ├── utils.py             <- Feature extraction and ohe transformation codes
│   ├── requirements.txt     <- List of Packages required to build backend
│   └── Dockerfile            <- Dockerfile for the backend
│
├── docker-compose.yml        <- Docker Compose file to run both services
├── render.txt                <- Instructions or notes for Render
└── github.txt                <- Instructions or notes for GitHub setup

```


## Running Locally
To run the application locally:
- change the url endpoint in config file 
``` python
# inside config.py file of /app

# prod_endpoint_url = 'https://backend-latest-6p8o.onrender.com/'
# fastapi_url = prod_endpoint_url

dev_endpoint_url = 'http://backend:8000/'
fastapi_url = dev_endpoint_url

```
Then build application using docker compose 
```docker
docker compose up --build
```
## Deployment on Render
### Deployment Overview

The application is deployed on Render, utilizing separate Docker images for both the frontend and backend services. The deployment process includes the following steps:

1. **Docker Image Creation**:
   - Docker images for both the frontend and backend are built, tagged appropriately, and pushed to Docker Hub.

2. **Web Service Setup**:
   - Individual web services are created on Render for the frontend and backend, ensuring a streamlined and efficient deployment process.

3. **Docker Compose Configuration**:
   - In the Docker Compose file, the dependency between the frontend and backend has been removed. This change allows for independent deployment of each service.

4. **Configuration Management**:
   - In the `config.py` file of frontend, the production endpoint(backend-webservice) is set to be used, while the development endpoint is commented out to prevent any accidental usage in the production environment.

5. **Repository Linking and Deployment**:
   - The tagged Docker images for the frontend and backend are pushed to the Docker repository. 
   - On Render, these repositories are separately linked and deployed to complete the deployment process.


### Access Application
https://frontend-o4ps.onrender.com/

### Access APIS
 URL: https://backend-latest-6p8o.onrender.com/
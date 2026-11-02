pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'ai-sign-language-recognition'
        AWS_REGION = 'us-east-1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE:$BUILD_NUMBER .'
            }
        }

        stage('Deploy to AWS') {
            steps {
                sh 'echo Deploying to AWS...'
                sh 'aws ecs update-service --cluster default --service ai-sign-language-service --force-new-deployment'
            }
        }
    }
}

pipeline {
  environment {
    registry = "suchithkumarch/scical"
    registryCredential = 'DockerHubCred'
    dockerImage = ''
  }
  agent any
  stages {
    stage('GIT CLONE') {
      steps {
        git 'https://github.com/suchithkumarch/Mask_detect_Live'
      }
    }
    stage('INSTALL REQUIREMENTS') {
      steps {
        echo "Installing requirements of the project"
        sh "pip3 install -r requirements.txt"
      }
    }
    stage('UNIT TEST') {
      steps {
        echo "Running the test cases of the project"
        sh "python3 source/test_model.py"
      }
    }

    stage('Building Docker image') {
      steps{
        script {
          echo "Building Docker image"
          dockerImage = docker.build registry + ":$BUILD_NUMBER"
        }
      }
    }
    stage('Deploy Docker Image') {
      steps{
        script {
          echo "Deploying Docker Image to " + registry
          docker.withRegistry( '', registryCredential ) {
            dockerImage.push('latest')
          }
        }
      }
    }
    stage('Remove unused/untagged docker images') {
      steps{
        sh "docker rmi $registry:$BUILD_NUMBER"
        sh "docker image prune"
      }
    }
    stage('Ansible') {
      steps{
        ansiblePlaybook becomeUser: null, colorized: true, disableHostKeyChecking: true, installation: 'Ansible',  playbook: 'Ansiblefile.yml', sudoUser: null
      }
    }    
  }
}


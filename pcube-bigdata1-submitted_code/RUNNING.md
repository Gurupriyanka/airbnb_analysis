# pcube-bigdata1

- Load all the input files and script files to Amazon S3

Airflow Setup:
- Create an EC2 instance with EMR Full Access and and IAM role with AWS EMR Full Access, EC2 Full Access and S3 Access
- Run the following commands in the AWS CLI of the created EC2 instance

sudo apt update

sudo apt install -y python3-pip

sudo apt install -y sqlite3

sudo apt-get install -y libpq-dev

sudo pip3 install virtualenv 

virtualenv venv 

source venv/bin/activate

pip3 install --upgrade awscli

pip3 install boto3

ip install "apache-airflow[postgres]==2.5.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.5.0/constraints-3.7.txt"

airflow db init

airflow users create -u airflow -f airflow -l airflow -r Admin -e youremail@gmail.com

mkdir /home/ubuntu/dags

cd airflow

vi airflow.cfg

change the following values

dags_folder = /home/ubuntu/dags

load_examples = False


- Make necessary changes in the airflow_code.py to reflect the correct input and script files location
- Load the airflow_code.py scripts to the dags_folder
- Launch the Airflow UI and trigger the DAG to run all the steps (EMR cluster creation, steps addition and execution and terminating the clusters)
- Check the status of the DAG steps under 'Graphs' tab

- Refresh the AWS Quicksight Dashboard

Image Classification and Collocation Analysis Setup and run:
- Load the listings dataset and images dataset in S3
- Create an EMR and add step to run pyspark_airbnb_image_analysis.py, PricePrediction.py, CollocationAnalysis.py
- Provide arguments as images folder and output folder
- Output files will be created in S3 buckets

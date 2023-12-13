import boto3
import logging
import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from time import sleep
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {"owner": "Airflow", "start_date": airflow.utils.dates.days_ago(2)}
aws_region = 'us-west-2'
dag = DAG(
    dag_id="airbnb_automation_dag", default_args=args, schedule_interval=None
)

client = boto3.client('emr', 
region_name='us-west-2',
aws_access_key_id='<your access key>',
aws_secret_access_key='<your secret key>')


def create_emr_cluster():
    client = boto3.client('emr', region_name=aws_region)
    
    cluster_id = client.run_job_flow(
        Name="Pyspark Airbnb",
        Instances={
            'InstanceGroups': [
                {
                    'Name': "Master",
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': 1,
                },
                {
                    'Name': "Slave",
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'CORE',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': 2,
                }
            ],
            
            'KeepJobFlowAliveWhenNoSteps': True,
            'TerminationProtected': False,
            
        },
        LogUri="s3://cmpt732-bigdataproject-pcube/airflow_logs/logsairflow.txt",
        ReleaseLabel='emr-6.2.0',
        BootstrapActions=[],
        VisibleToAllUsers=True,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        Applications=[{'Name': 'Spark'}, {'Name': 'Hive'}]
    )
    
    print("The cluster started with cluster id : {}".format(cluster_id))
    return cluster_id

       
def add_step_emr(cluster_id,jar_file,step_args):
    print("The cluster id : {}".format(cluster_id))
    print("The step to be added : {}".format(step_args))
    response = client.add_job_flow_steps(
    JobFlowId=cluster_id,
    Steps=[
        {
          'Name': 'pyspark_airbnb',
          'ActionOnFailure':'CONTINUE',
          'HadoopJarStep': {
        'Jar': jar_file,
        'Args': step_args
    }
    },
    ]
    )
    print("The emr step is added")
    return response['StepIds'][0]
    
def wait_for_step_to_complete(cluster_id, step_id):
    print("The cluster id : {}".format(cluster_id))
    print("The emr step id : {}".format(step_id))
    client = boto3.client('emr', region_name=aws_region)
    while True:
        try:
          status=get_status_of_step(cluster_id,step_id)
          if status =='COMPLETED':
            break
          else:
            print("The step is {}".format(status))
            sleep(4)

        except Exception as e:
          logging.info(e)
          
          
def get_status_of_step(cluster_id,step_id):
  response = client.describe_step(
    ClusterId=cluster_id,
    StepId=step_id
  )
  return response['Step']['Status']['State']
  
  
def terminate_cluster(cluster_id):
    try:
        client.terminate_job_flows(JobFlowIds=[cluster_id])
        logger.info("Terminated cluster %s.", cluster_id)
    except ClientError:
        logger.exception("Couldn't terminate cluster %s.", cluster_id)
        raise
    
# DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with dag:
    # Create an EMR cluster
    create_cluster = PythonOperator(
        task_id='create_emr_cluster',
        python_callable=create_emr_cluster,
        dag=dag,
    )

    # Add all PySpark steps to the EMR cluster
    add_pyspark_step_1 = PythonOperator(
        task_id='pyspark_airbnb_step_1',
        python_callable=add_step_emr,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','command-runner.jar',[ 'spark-submit',
                '--master', 'yarn',
                '--deploy-mode', 'client',
                's3://cmpt732-bigdataproject-pcube/Scripts/Enhancing-ETL.py s3://cmpt732-bigdataproject-pcube/Dataset/listings_canada/ s3://cmpt732-bigdataproject-pcube/Output/op-5' ]],
        dag=dag, 
    )


    # Sensor to wait for all PySpark steps completion
    wait_for_pyspark_step_1 = PythonOperator(
        task_id='wait_for_steps_completion_1',
        python_callable=wait_for_step_to_complete,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','{{ ti.xcom_pull("pyspark_airbnb_step_1")}}'],
        dag=dag, 
    )
    
    # Add all PySpark steps to the EMR cluster
    add_pyspark_step_2 = PythonOperator(
        task_id='pyspark_airbnb_step_2',
        python_callable=add_step_emr,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','command-runner.jar',[ 'spark-submit',
                '--master', 'yarn',
                '--deploy-mode', 'client',
                's3://cmpt732-bigdataproject-pcube/Scripts/pyspark_airbnb.py s3://cmpt732-bigdataproject-pcube/Dataset/listings_canada/ s3://cmpt732-bigdataproject-pcube/Dataset/worldcities/worldcities/ s3://cmpt732-bigdataproject-pcube/Output/Correlation_price.csv s3://cmpt732-bigdataproject-pcube/Output/Correlation_ratings.csv s3://cmpt732-bigdataproject-pcube/Output/Correlation_text' ]],
        dag=dag, 
    )


    # Sensor to wait for all PySpark steps completion
    wait_for_pyspark_step_2 = PythonOperator(
        task_id='wait_for_steps_completion_2',
        python_callable=wait_for_step_to_complete,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','{{ ti.xcom_pull("pyspark_airbnb_step_2")}}'],
        dag=dag, 
    )
    
    # Add all PySpark steps to the EMR cluster
    add_pyspark_step_3 = PythonOperator(
        task_id='pyspark_airbnb_step_3',
        python_callable=add_step_emr,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','command-runner.jar',[ 'spark-submit',
                '--master', 'yarn',
                '--deploy-mode', 'client',
                's3://cmpt732-bigdataproject-pcube/Scripts/SentimentAnalysis.py s3://cmpt732-bigdataproject-pcube/Dataset/reviews_canada/ s3://cmpt732-bigdataproject-pcube/Output/sentiment_analysis/' ]],
        dag=dag, 
    )


    # Sensor to wait for all PySpark steps completion
    wait_for_pyspark_step_3 = PythonOperator(
        task_id='wait_for_steps_completion_3',
        python_callable=wait_for_step_to_complete,
        op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}','{{ ti.xcom_pull("pyspark_airbnb_step_3")}}'],
        dag=dag, 
    )

    # PythonOperator to terminate EMR cluster
    terminate_emr_cluster = PythonOperator(
      task_id='terminate_emr_cluster',
      python_callable=terminate_cluster,
      op_args=['{{ ti.xcom_pull("create_emr_cluster")["JobFlowId"]}}'],
      dag=dag, 
    )

    # Define the DAG execution flow
    create_cluster >> add_pyspark_step_1 >> wait_for_pyspark_step_1 >> add_pyspark_step_2 >> wait_for_pyspark_step_2 >> add_pyspark_step_3 >> wait_for_pyspark_step_3 >> terminate_emr_cluster
    

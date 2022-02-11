import os

SOURCE_FILE='C:\\Users\\momob\\Documents\\frontend\\33000-BORDEAUX_nettoye.csv'
SOURCE_FILE_NAME='33000-BORDEAUX_nettoye.csv'
PORT_HADOOP='2222'
LOCAL_FOLDER='C:\\Users\\momob\\Documents\\frontend\\'
KEY_FILE_NAME='bigdatanew.pem'
PUBLIC_IPV4_DNS='ec2-3-86-229-237.compute-1.amazonaws.com'

print("copier le fichier source csv de la machine physique vers la machine virtuelle qui contient cluster hadoop par scp (secure copy)")
os.system('scp -P '+PORT_HADOOP+' '+ SOURCE_FILE + ' root@127.0.0.1:/root;')
print("mettre le fichier csv sur hdfs par ssh")
os.system('ssh -t root@127.0.0.1 -p '+PORT_HADOOP+' "hdfs dfs -put /root/'+SOURCE_FILE_NAME+' /input;"')
print("recuperer le fichier csv du hdfs sur la machine virtuelle hadoop")
os.system('ssh -t root@127.0.0.1 -p '+PORT_HADOOP+' "hdfs dfs -get /input/'+SOURCE_FILE_NAME+' /root/middleRep;"')
print("copier le fichier de la machine virutelle hadoop vers la machine physique qui est windows dans notre cas")
os.system('scp -P '+PORT_HADOOP+' root@127.0.0.1:/root/middleRep/'+SOURCE_FILE_NAME+' '+LOCAL_FOLDER+'')
print("faire une configuration aws")
os.system('aws configure')
print("copier le fichier de la machine physique vers l'instance EC2 ")
os.system('scp -i ./'+KEY_FILE_NAME+' .\\'+SOURCE_FILE_NAME+' ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\preprocess.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\predict.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
os.system('scp -i ./'+KEY_FILE_NAME+' .\\utils.py ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
#os.system('scp -i ./'+KEY_FILE_NAME+' .\\requirements.txt ec2-user@'+PUBLIC_IPV4_DNS+':/home/')
print("copier le fichier de l'instance EC2 vers la machine physique ")
os.system('scp -i ./'+KEY_FILE_NAME+' ec2-user@'+PUBLIC_IPV4_DNS+':/home/predict.json '+LOCAL_FOLDER)
print("se connecter en ssh sur l'instance EC2")
os.system('ssh -i "'+KEY_FILE_NAME+'" ec2-user@'+PUBLIC_IPV4_DNS+'')
#python3 -m venv ./ml
#source ./ml/bin/activate
#pip3 install -r requirements.txt
#python preprocess.py
#python predict.py




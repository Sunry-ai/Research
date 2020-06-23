# ToyProgramForOKApp

												作成者:王　啓航
# 1.背景
　本プログラムはOK-AppというRESTAPIの使い方法を説明するために作成された物です。
# 2.目的
　簡単のために，SVMを用いたMNIST(手書き数字)識別のトーイプログラムを作りました。タスクとしてはこのローカルにあるプログラムをOK-AppでRUNすることです。まず，フォルダーの中身について説明します。構造としては以下です。

```.
├── MNIST_Classifiation_SVM.ipynb
├── dataset_jpg
│   ├── N_ID.jpg
│   └── ...
├── MNIST_SVM
├── svm_model.py
```

一つずつ説明します。
 1. 「MNIST_Classifiation_SVM.ipynb」:訓練関連のプログラムは「に入っています(ハイパーパラメータは特にチューニングしていません)
 2. 「dataset」:MNISTデータをこのフォルダーに保存した。	
 3. 「N_ID.jpg」の形で図を保存した。Nは数字、IDは図の番号を示しています。
 4. 「MNIST_SVM」:joblibを用いて保存した訓練済みのモデルです
 5. 「svm_model.py」:Ok-Appに適用するモデルです。
#

# 3.手順(Docker container でInferまで)

## 1. 仮想環境作成

```
python -m venv MNIST_SVM_ENV
```

ここで**MNIST_SVM_ENV**は自分で環境名を定義します。(注意する必要なのは、ここのpythonは3.6以上、pipは20.1.1以上に更新する必要があります。)
　 
```
source MNIST_SVM_ENV/bin/activate
```

でMNIST_SVM_ENVの環境に入ります。

## 2. app runnerはコマンドラインから呼び出す前提のため、okapprunnerをインストールします。

```
$ pip install -e "git+https://gitlab.ai-team.datasuite-devel.com/mlplatform/ok-app-runner.git#egg=okapprunner&subdirectory=okapprunner"
```
## 3. 初期化
まずフォルダーを作ります。
```
mkdir dev_dockerfile
```

OK-Appのmappingはtext,image,feature,image2image,
text2raw,raw2raw,image2rawの計7種類存在します。MNISTの例では画像データをサーバーにアップロードすれば，数字を返すのでimage2rawに当てはまります。
以下のコードを使用し、初期化することができます。

```
cd dev_dockerfile
``` 

```
python -m okapprunner.runner init --class-name MNIST_SVM --mapping image2raw
```

このプログラムはSVMを用いたMNISTの予測であるため、名前を**MNIST_SVM**にした。他の名前を指定してもできます。
　
    
```.
├── Dockerfile
├── mnist_svm
│   ├── mnist_svm.py
│   └── setup.py    
│   └── requirements.txt
├── bin/*
```

**Dockerfile**
      Dockerのimageを作成する時に必要なファイル，デフォルトのが生成されるが、需要に応じて変更する必要があります。(ここは変更しますが、後ほど説明します)

**MNIST_SVM/**
	 -ここ名前はinitモードで指定した--class-nameで指定された値に応じて変更されます。mnist_svm/mnist_svm.pyの中身を**ローカルのsvm_model.py**に置き換え、アプリケーションの作成を行ってください。
     -指定したmappingの内容により、XXXX.pyに必要な関数が変わる。trainingする場合は、fit(X, y), score(X, y)関数は必須となります。今回のケースでは要らないです。行わない場合はmappingの値によりpredict(X), もしくはpredict_proba(X)が必要となります。 詳しい内容はDOCUMENTのmappingの項目をご覧ください。 **本プログラムはimage2rawであるため、predict(X)を使用します。**
     -requirement.txtに必要なライブラリーを記入します。今回のToy Programeでは下記のライブラリーが必要のため、 
``` 
scikit-learn==0.21.2
numpy==1.16.2
ddtrace==0.26.0
opencv-python==4.2.0.34
joblib==0.15.1
``` 

モデルのpathを以下のように置き換えます。Dockerfileを生成する際に、訓練済みモデルのpathを変更する必要があります。

変更前
```    
      # Load Model
        self.model=joblib.load("MNIST_SVM")
```
変更後
```
     # Load Model
        self.model=joblib.load("/opt/mlpf_app/mnist_svm/MNIST_SVM")
```
**bin/**
    bin/trainがtrain時に実行されるコマンド、bin/appがserving時に実行されるコマンドです。基本的に変更はいりません。

## 4. APIを立ち上げ

    appモードで起動することで、REST APIとしてmodelをserveすることができます。手順は以下のようです。
    (1)モデルの(.pyファイル)のパスに入ります。
    (2)以下のコマンドを入力します。

```
$python -m okapprunner.runner app --mapping image2raw --module_path mnist_svm.MNIST_SVM
```

注意: ここのportは1234ですが、もし **-p 1234**指定しないなら、デフォルトは80です(一般的にはデフォルトを使用します)。**mnist_svm.MNIST_SVM**はmnist_svmのMNIST_SVM moduleを使用します。今回はmoduleを使用するため **--module_path**と記述します。またもしmodelを使用するならここは **--model_path**とします。


## 5. ローカル環境でのInfer
**注意** この場合、訓練ずみモデルのパスは訓練前のを使用します。

入力をbase64でエンコードしないといけないので、エンコードされた画像を入力する必要があります。以下の２つの方法で画像をbase64でエンコードできます。<br>
(1) -dで指定する方法<br>
``` 
(echo -n '{"sample": "'; bas64 画像名; echo '"}') | curl --header "Content-type: application/json" -X POST "http://localhost:1234/infer" -d @- 
```  
    
この方法を使用する際に、必ず--header "Content-type: application/json" を付ける必要があります。
    
(2) -Fで指定する方法
    
```
curl http://localhost:1234/infer -F sample=@7_1761.jpg
```
	
この方法を使用する際に--header "Content-type: application/json" を **付けないでください**。注意:入力はリストの形ですので、X[0]がbase64でエンコードされる画像です。

## 6. Dockerfileの編集

初期化で生成されるDockerfileは以下です。	
	
```
	FROM python:3.7

	WORKDIR /opt/mlpf_app

	RUN pip install -e "git+https://oauth2:tGg67Vti8YQM6QvFRDp5@gitlab.ai-team.datasuite-devel.com/mlplatform/ok-app-runner.git@master#egg=okapprunner&subdirectory=okapprunner"

	COPY mnist_svm /opt/mlpf_app/mnist_svm

	RUN pip install -e mnist_svm

	COPY bin /opt/mlpf_app/bin
	RUN chmod +x bin/*
	RUN mkdir /opt/mlpf_app/application
	COPY config.yaml /opt/mlpf_app/application/config.yaml
	RUN mkdir /opt/mlpf_app/application/datasets
	RUN mkdir /opt/mlpf_app/application/models
	RUN chmod +x application/*
```

この場合、mnist_svmのフォルダーとbinが移動されるのです。ubuntu環境のcontainerを生成するには不十分です。**FROM python:3.7を消して,以下の内容を加えます。**
	
```
	FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

	ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive
	ENV TZ Asia/Tokyo

	RUN apt-get update --fix-missing && apt-get install -y cmake build-essential gcc g++ wget bzip2 ca-certificates libglib2.0-dev libxext6 libsm6 libxrender1 git mercurial subversion lightdm pkg-config libavcodec-dev libavformat-dev libswscale-dev vim

	RUN apt-get install -y software-properties-common
	RUN add-apt-repository -y ppa:deadsnakes/ppa
	RUN apt-get update && apt-get install -y python3.6 python3.6-dev
	RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 0
	RUN rm -rf /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python
	RUN apt-get install -y python3-pip
	RUN pip3 install --upgrade pip
```

## 7. Docker imageの作成

Dockerfileのpathに入り、以下のコマンドを入力し、imageを作成します。
```
docker build -t mlapi/myapp .	
```
(少し時間がかかります。(5min程度))

## 8. Docker containerの作成

次にコマンドを入力すると，既存のimageを確認できます。

```
    docker images　ls
```
	
mlapi/myappの存在を確認できましたら、次に以下のコマンドを入力し、containerを立ち上げます。
    
```
    docker run -it -p 1234:80 mlapi/myapp　
```

ここで **-p 1234:80**は「container内部の80portが外部の1234と対応します。」と意味します。

## 9. 試しにcontainer上でinferします

まず、container内部でAPIを立ち上げます。Dockerfile通りに、minist_svmが ``/opt/mlpf_app/mnist_svm ``にあります。このパスに入ります。次に、次のコマンドを入力し、APIを起動。
    
```
    $python -m okapprunner.runner app --mapping image2raw --module_path mnist_svm.MNIST_SVM
```
　  
最後に、コマンドライン(Docker containerではなく、ローカルの方)で予測したい画像のパスに入り，次のinferコマンドを記入します。
    
```
    curl http://localhost:1234/infer -F sample=@7_1761.jpg
``` 
   
   ここの**7_1761.jpg**は入力画像です。7はこの画像の数値です。
   そうすると、以下返し値を確認できます。
   
```.
	{
  "meta": {},
  "data": {
    "type": "inference-result",
    "attributes": {
      "result": [
        {
          "number": "7"
        }
      ]
    }
  }
```
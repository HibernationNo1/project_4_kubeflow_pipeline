

`Ubuntu 18.04`



## install

### requirements

1. **install NVIDIA driver**

   ```
   $ nvidia-smi
   ```

   



### docker

[공식 문서](https://docs.docker.com/engine/install/ubuntu/)

**Install using the repository**

1. install docker의 prerequisite packge

   ```
   $ sudo apt-get install \
       ca-certificates \
       curl \
       gnupg \
       lsb-release
   ```

   

2. GPH key추가

   ```
   $ sudo mkdir -p /etc/apt/keyrings
   $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

   

3. repository를 follow하도록 설정

   ```
   $ echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

   > arm기반의 cpu인 경우 위 명령어 대신 다른 명령어 사용(검색하기)

   

4. install Docker Engine (최신 version)

   ```
   $ sudo apt-get update
   $ sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

   > 특정 version의 docker engine을 install하고자 한다면 공식 문서 참고

5. Create required directories

   ```
   $ sudo mkdir -p /etc/systemd/system/docker.service.d
   ```

   

6. Create daemon json config file

   ```
   $ sudo tee /etc/docker/daemon.json <<EOF
   {
     "exec-opts": ["native.cgroupdriver=systemd"],
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "100m"
     },
     "storage-driver": "overlay2"
   }
   EOF
   ```

   ```
   $ sudo systemctl daemon-reload 
   ```

   

7. check

   ```
   $ sudo docker run hello-world
   ```

   `Hello from Docker!` 이 포함된 출력문이 나오면 된것

   

8. 권한 설정

   root user가 아닌, host의 기본 user에게도 권한을 주기 위해 

   새로운 터미널 띄운 후 

   ```
   $ sudo usermod -a -G docker $USER
   $ sudo systemctl restart docker
   $ sudo systemctl enable docker
   ```

   이후 logout(또는 reboot)후 다시 login

   ```
   $ docker ps
   ```

   

#### Install Mirantis cri-dockerd

[공식 문서](https://computingforgeeks.com/install-mirantis-cri-dockerd-as-docker-engine-shim-for-kubernetes/)

Docker 엔진의 경우 shim 인터페이스가 필요

Kubernetes용 Docker Engine shim으로 **Mirantis cri-dockerd** 설치

> Kubernetes가 v1.20 이후 컨테이너 런타임으로 Docker를 더 이상 사용하지 않지만 Docker 생성 이미지는 항상 그래왔듯이 모든 런타임과 함께 Kubernetes 클러스터에서 계속 작동된다.
>
> **cri-dockerd**를 사용하면 Docker 엔진이 CRI를 준수할 수 있으며, 기본 제공 dockershim에서 외부 dockershim으로 전환하기만 하면 Kubernetes에서 계속 사용할 수 있다.

Mirantis cri-dockerd CRI 소켓 파일 경로는 `/run/cri-dockerd.sock` (Kubernetes 클러스터를 구성할 때 사용)

1. get the latest release version

   ```
   $ VER=$(curl -s https://api.github.com/repos/Mirantis/cri-dockerd/releases/latest|grep tag_name | cut -d '"' -f 4|sed 's/v//g')
   $ echo $VER
   ```

2. download the archive file from [Github cri-dockerd releases](https://github.com/Mirantis/cri-dockerd/releases) page.

   ```
   $ wget https://github.com/Mirantis/cri-dockerd/releases/download/v${VER}/cri-dockerd-${VER}.amd64.tgz
   tar xvf cri-dockerd-${VER}.amd64.tgz
   ```

   Move `cri-dockerd` binary package to `/usr/local/bin` directory

   ```
   $ sudo mv cri-dockerd/cri-dockerd /usr/local/bin/
   ```

3. Validate successful installation

   ```
   $ cri-dockerd --version
   ```

   ```
   cri-dockerd 0.2.5 (10797dc)
   ```

4. Configure systemd

   ```
   $ wget https://raw.githubusercontent.com/Mirantis/cri-dockerd/master/packaging/systemd/cri-docker.service
   $ wget https://raw.githubusercontent.com/Mirantis/cri-dockerd/master/packaging/systemd/cri-docker.socket
   $ sudo mv cri-docker.socket cri-docker.service /etc/systemd/system/
   $ sudo sed -i -e 's,/usr/bin/cri-dockerd,/usr/local/bin/cri-dockerd,' /etc/systemd/system/cri-docker.service
   ```

5. Start and enable the services

   ```
   $ sudo systemctl daemon-reload
   $ sudo systemctl enable cri-docker.service
   $ sudo systemctl enable --now cri-docker.socket
   ```

6. Confirm the service is now running

   ```
   $ systemctl status cri-docker.socket
   ```

   ```
   ● cri-docker.socket - CRI Docker Socket for the API
        Loaded: loaded (/etc/systemd/system/cri-docker.socket; enabled; vendor preset: enabled)
        Active: active (listening) since Tue 2022-09-20 14:01:38 KST; 11s ago
      Triggers: ● cri-docker.service
        Listen: /run/cri-dockerd.sock (Stream)
         Tasks: 0 (limit: 76823)
        Memory: 116.0K
        CGroup: /system.slice/cri-docker.socket
   
   Sep 20 14:01:38 ubuntu systemd[1]: Starting CRI Docker Socket for the API.
   Sep 20 14:01:38 ubuntu systemd[1]: Listening on CRI Docker Socket for the API.
   ```

   

#### NVIDIA DOCKER

docker contianer안에서 GPU를 사용하기 위해선 필수

1. Setting up NVIDIA Container Toolkit

   ```
   $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
         && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
         && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
               sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
               sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

   > `ubuntu18.04/$(ARCH)` 떠도 20.04에서 정상 작동

2. install nvidia-docker2

   ```
   $ sudo apt-get update
   $ sudo apt-get install -y nvidia-docker2
   ```

   ```
   *** daemon.json (Y/I/N/O/D/Z) [default=N] ? y
   Installing new version of config file /etc/docker/daemon.json ...
   ```

3. Restart the Docker daemon 

   ```
   $ sudo systemctl restart docker
   ```

   

   confirm : 기본 CUDA container 실행

   ```
   $ sudo docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
   ```

   > cuda와 ubuntu version에 대한tag는 [docker hub-nvidia](https://hub.docker.com/r/nvidia/cuda/tags)에서 검색 후 결정

4. edit daemon

   ```
   $ sudo vi /etc/docker/daemon.json
   ```

   아래내용 추가

   ```
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

   ```
   $ sudo systemctl daemon-reload 
   ```

   

   



### minikube

[공식](https://minikube.sigs.k8s.io/docs/start/)

CPU 2core 이상, Memory 2GB이상, Disk : 20GB이상

```
$ curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
$ sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

check

```
$ minikube version
```



#### kubectl

[공식](https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-linux/)

특정 release 다운로드(1.20.13) (release확인은 [여기](https://kubernetes.io/releases/) 에서)

```
$ sudo curl -LO https://dl.k8s.io/release/v1.22.13/bin/linux/amd64/kubectl
```

바이너리 검증

```
$ curl -LO "https://dl.k8s.io/v1.22.13/bin/linux/amd64/kubectl.sha256"
$ echo "$(<kubectl.sha256)  kubectl" | sha256sum --check
```

> 검증 성공시 아래처럼 출력
>
> ```
> kubectl: OK
> ```

install kubectl

```
$ sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

check

```
$ kubectl version --client
```

> ```
> Client Version: version.Info{Major:"1", Minor:"24", GitVersion:"v1.24.2", GitCommit:"f66044f4361b9f1f96f0053dd46cb7dce5e990a8", GitTreeState:"clean", BuildDate:"2022-06-15T14:22:29Z", GoVersion:"go1.18.3", Compiler:"gc", Platform:"linux/amd64"}
> ```
>
> 위 처럼 떠도 정상 (kubenetes server와 client의 version이 모두 출력하는 과정에서, host에서 kubenetes server를 생성하지 않았기 때문에 뜨는 문구)
>
> - `bash: /usr/bin/kubectl: No such file or directory` 라는 문구가 뜨면
>
>   ```
>   $ bash
>   ```
>   
>   또는 새 터미널에서 실행
>





#### start minikube

confirm host

```
$ sudo vim /etc/hosts
```

아래 두 개가 제대로 존재하는지 확인

```
127.0.0.1       host.minikube.internal
192.168.0.107   control-plane.minikube.internal
```



```
$ minikube start --driver=none \
  --kubernetes-version=v1.22.13 \
  --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key \
  --extra-config=apiserver.service-account-issuer=kubernetes.default.svc
```

- `--kubernetes-version ` : kubectl 설치 시 특정한 version
- `--extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key` : kubeflow를 사용하기 위한 flag
- `--extra-config=apiserver.service-account-issuer=kubernetes.default.svc`  : kubeflow를 사용하기 위한 flag

> error
>
> - ```
>   Exiting due to PROVIDER_NONE_NOT_FOUND: The 'none' provider was not found: running the 'none' driver as a regular user requires sudo permissions
>   ```
>
>   sudo 붙여서 실행
>
> - ```
>   Exiting due to HOST_JUJU_LOCK_PERMISSION: Failed to save config: failed to acquire lock for /home/ainsoft/.minikube/profiles/minikube/config.json: {Name:mk2998bbe62a1ef4b160001f97b8d3cac88d028d Clock:{} Delay:500ms Timeout:1m0s Cancel:<nil>}: unable to open /tmp/juju-mk2998bbe62a1ef4b160001f97b8d3cac88d028d: permission denied
>   ```
>
>   해결방법 
>
>   ```
>   $ sudo rm -rf /tmp/juju-mk*
>   $ sudo rm -rf /tmp/minikube.*
>   ```
>
>   



check

```
$ minikube status
```

```
minikube
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```



```
$ kubectl get namespace
```

> error
>
> ```
> error: unable to read client-key /home/ainsoft/.minikube/profiles/minikube/client.key for minikube due to open /home/ainsoft/.minikube/profiles/minikube/client.key: permission denied
> ```
>
> 해결방법 : 권한 설정
>
> ```
> sudo chown -R $HOME/.minikube
> ```
>
> 이래도 안되면
>
> ```
> sudo chown -R $USER $HOME/.kube
> ```
>
> 도 추가

```
kubectl get pods -A
```



#### nvidia-device-plugin

1. install

   ```
   $ kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
   ```

   

   check

   ```
   $ kubectl get pod -A | grep nvidia
   ```

   > ```
   > kube-system   nvidia-device-plugin-daemonset-rs69d   1/1     Running   0          48s
   > ```

   

   ```
   $ kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
   ```

   > ```
   > NAME     GPU
   > ubuntu   1
   > ```
   >
   > 위 처럼 1이 보여야 한다. (몇 초 가량 지나야함)

2. check use GPU at pod

   create pod

   ```
   $ vi gpu-container.yaml
   ```

   ```
   apiVersion: v1
   kind: Pod
   metadata:
     name: gpu
   spec:
     containers:
     - name: gpu-container
       image: nvidia/cuda:11.3.1-runtime-ubuntu20.04
       command:
         - "/bin/sh"
         - "-c"
       args:
         - nvidia-smi && tail -f /dev/null
       resources:
         requests:
           nvidia.com/gpu: 1
         limits:
           nvidia.com/gpu: 1
   ```

   - `nvidia/cuda:10.2-runtime` : 알맞는 cuda version명시해줘야 함
   - ``spec.resources.requests` 와 `spec.resources.limits` 에 `nvidia.com/gpu` 를 포함해야 pod 내에서 GPU 사용이 가능` ★★★

   ```
   $ kubectl create -f gpu-container.yaml
   $ kubectl get pod gpu -n default
   ```
   
   ```
   NAME   READY   STATUS              RESTARTS   AGE
   gpu    0/1     ContainerCreating   0          90s
   ```
   
   > `STATUS : Runniing` 확인 후 아래 명령어 실행
   
   ```
   $ kubectl logs gpu
   ```
   
   
   
   ```
   Thu Aug 25 00:45:45 2022       
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |===============================+======================+======================|
   |   0  GeForce RTX 208...  On   | 00000000:01:00.0 Off |                  N/A |
   |  0%   40C    P8    12W / 300W |   2672MiB / 11019MiB |      0%      Default |
   +-------------------------------+----------------------+----------------------+
                                                                                  
   +-----------------------------------------------------------------------------+
   | Processes:                                                       GPU Memory |
   |  GPU       PID   Type   Process name                             Usage      |
   |=============================================================================|
   +-----------------------------------------------------------------------------+
   ```



### kubeflow

#### **kustomize** 

[여기](https://github.com/kubernetes-sigs/kustomize/) 에서 현재 k8s version에 맞는 kustomize version을 확인하고 download binary

```
$ kubectl version
```

> kustomize 3.2.0에 알맞는 version확인

[여기](https://github.com/kubernetes-sigs/kustomize/releases/tag/v3.2.0)의 **Asset** 아래 `kustomize_3.2.0_darwin_amd64` 의 링크 복사 (arm이면 arm꺼 복사)

> 링크 없어지면 [releases](https://github.com/kubernetes-sigs/kustomize/releases?page) 에서 3.2.0 찾은 후 진행

```
$ sudo wget https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
```

> - 4.2.0 설치 시 (**아직까진  kubeflow가 3.2.0외의 version과는 호환되지 않고 있음**)
>
>   releases에서 4.2.0찾은 후 `kustomize_v4.2.0_linux_amd64.tar.gz` 복사 
>
>   ```
>   sudo wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv4.2.0/kustomize_v4.2.0_linux_amd64.tar.gz
>   ```
>
>   
>
>   압축 풀고 진행
>
>   ```
>   $ gzip -d kustomize_v4.2.0_linux_amd64.tar.gz
>   $ tar -xvf kustomize_v4.2.0_linux_amd64.tar
>   ```

file의 mode 변경 (실행 가능하도록)

```
$ sudo chmod +x kustomize_3.2.0_linux_amd64
```

압축 풀고 file위치 변경

```
$ sudo mv kustomize_3.2.0_linux_amd64 /usr/local/bin/kustomize
```

check(새 terminal 열고)

```
$ kustomize version
```

```
Version: {KustomizeVersion:3.2.0 GitCommit:a3103f1e62ddb5b696daa3fd359bb6f2e8333b49 BuildDate:2019-09-18T16:26:36Z GoOs:linux GoArch:amd64}
```

> uninstall : `sudo rm /usr/local/bin/kustomize`
>
> - 4.2.0설치 시 `kustomize_3.2.0_linux_amd64` 라는 file 대신 `kustomize` 라는 file 존재
>
>   ```
>   $ sudo chmod +x kustomize
>   $ sudo mv kustomize /usr/local/bin/kustomize
>   $ kustomize version
>   ```





#### **kubeflow**

1. git clone [kubeflow/manifests](https://github.com/kubeflow/manifests)

   ```
   $ cd ~/hibernation			# git clone할 임의의 위치
   $ git clone https://github.com/kubeflow/manifests.git
   $ cd manifests
   ```

   > ```
   > $ git checkout tags/v1.4.0 
   > ```
   >
   > 위 명령어를 통해 특정 version으로 checkout하면 `manifests/apps/pipeline/upstream/env/` 의 cert-manager dir이 사라지는  현상 발생 

   1. automatically install

      ```
      $ while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
      ```

   2. manually install

      kubeflow의 individual components install ([github](https://github.com/kubeflow/manifests) 에 다 있음. 가능하면 해당 link에서 보고 install)

      > 각각 yaml file을 build이후 kubectl apply -f를 진행하게 되는, 이는 모두 순서대로 해야한다. 특정 kubeflow version을 설치한다면, 대한 version에 대한 tag로 이동하여 해당 version에 맞는 명령어를 입력해야 한다.

      1. cert-manager

         ```
         $ kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
         $ kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -
         ```

         check

         ```
         $ kubectl get pod -n cert-manager
         ```

      2. istio

         ```
         $ kustomize build common/istio-1-14/istio-crds/base | kubectl apply -f -
         $ kustomize build common/istio-1-14/istio-namespace/base | kubectl apply -f -
         $ kustomize build common/istio-1-14/istio-install/base | kubectl apply -f -
         ```

         > kubeflow version에 따라 istio의 version이 다를 수 있으니 확인할 것

         ```
         $ kubectl get pod -n istio-system 
         ```

      3. Dex

         ```
         $ kustomize build common/dex/overlays/istio | kubectl apply -f -
         ```

      4. OIDC AuthService

         ```
         $ kustomize build common/oidc-authservice/base | kubectl apply -f -
         ```

      5. Knative

         > 설치 안됨 
         >
         > ```
         > unable to recognize "STDIN": no matches for kind "PodDisruptionBudget" in version "policy/v1"
         > unable to recognize "STDIN": no matches for kind "PodDisruptionBudget" in version "policy/v1"
         > ```
         >
         > 

         ```
         $ kustomize build common/knative/knative-serving/overlays/gateways | kubectl apply -f -
         $ kustomize build common/istio-1-14/cluster-local-gateway/base | kubectl apply -f -
         ```

      6. Kubeflow Namespace

         ```
         $ kustomize build common/kubeflow-namespace/base | kubectl apply -f -
         ```

         check

         ```
         $ kubectl get namespace   # Kubeflow라는 namespace생성되어야함
         ```

      7. Kubeflow Roles

         ```
         $ kustomize build common/kubeflow-roles/base | kubectl apply -f -
         ```

      8. Kubeflow Istio Resources

         ```
         $ kustomize build common/istio-1-14/kubeflow-istio-resources/base | kubectl apply -f -
         ```

         > kubeflow version에 따라 istio의 version이 다를 수 있으니 확인할 것

      9. Kubeflow Pipelines

         ```
         $ kustomize build apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user | kubectl apply -f -
         ```

         > If your container runtime is not docker, use pns executor instead:
         >
         > ```
         > $ kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user-pns | kubectl apply -f -
         > ```

         만약 아래와 같은 error가 떳다면

         ```
         unable to recognize "STDIN": no matches for kind "CompositeController" in version "metacontroller.k8s.io/v1alpha1"
         ```

         위 설치 명령어 다시 입력

      10. KServe

          Install the KServe component

          ```
          $ kustomize build contrib/kserve/kserve | kubectl apply -f -
          ```

          > ```
          > anable to recognize "STDIN": no matches for kind "ClusterServingRuntime" in version "serving.kserve.io/v1alpha1"
          > ```
          >
          > 가 뜬다면 위 명령어 한번 더 입력

          Install the Models web app

          ```
          $ kustomize build contrib/kserve/models-web-app/overlays/kubeflow | kubectl apply -f -
          ```

      11. Katib

          ```
          $ kustomize build apps/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -
          ```

      12. Central Dashboard

          ```
          $ kustomize build apps/centraldashboard/upstream/overlays/kserve | kubectl apply -f -
          ```

      13. Admission Webhook

          ```
          $ kustomize build apps/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -
          ```

      14. Notebooks

          Install the Notebook Controller official Kubeflow component

          ```
          $ kustomize build apps/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
          ```

          Install the Jupyter Web App official Kubeflow component

          ```
          # kustomize build apps/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -
          ```

      15. Profiles + KFAM

          ```
          $ kustomize build apps/profiles/upstream/overlays/kubeflow | kubectl apply -f -
          ```

      16. Volumes Web App

          ```
          $ kustomize build apps/volumes-web-app/upstream/overlays/istio | kubectl apply -f -
          ```

      17. Tensorboard

          Install the Tensorboards Web App official Kubeflow component

          ```
          $ kustomize build apps/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
          ```

          Install the Tensorboard Controller official Kubeflow component

          ```
          $ kustomize build apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -
          ```

      18. training operator

          ```
          $ kustomize build apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
          ```

      19. User Namespace

          ```
          $kustomize build common/user-namespace/base | kubectl apply -f -
          ```

      

2. 모든 pod 구동

   ```
   $ kubectl get po -A -w
   ```

   > 길게는 몇십분까지 걸림







## add user

dashboard에 user를 추가하기 위해서는 cm dex를 수정해야 한다.

1. **check dex**

   dex는 namespace auth에 있음

   ```
   $ kubectl -n auth get cm dex -o yaml
   ```

   ```yaml
   apiVersion: v1
   data:
     config.yaml: |
       issuer: http://dex.auth.svc.cluster.local:5556/dex
       storage:
         type: kubernetes
         config:
           inCluster: true
       web:
         http: 0.0.0.0:5556
       logger:
         level: "debug"
         format: text
       oauth2:
         skipApprovalScreen: true
       enablePasswordDB: true
       staticPasswords:
       - email: user@example.com
         hash: $2y$12$4K/VkmDd1q1Orb3xAt82zu8gk7Ad6ReFR4LCP9UeYE90NLiN9Df72
         # https://github.com/dexidp/dex/pull/1601/commits
         # FIXME: Use hashFromEnv instead
         username: user
         userID: "15841185641784"
       staticClients:
       # https://github.com/dexidp/dex/pull/1664
       - idEnv: OIDC_CLIENT_ID
         redirectURIs: ["/login/oidc"]
         name: 'Dex Login Application'
         secretEnv: OIDC_CLIENT_SECRET
   ... 이하 생략
   
   ```

   위의 `staticPasswords` 에 아래 4가지를 추가해야 한다.

   ```
   - email: winter4958@gmail.com
     hash: $2a$12$lRDeywzDl4ds0oRR.erqt.b5fmNpvJb0jdZXE0rMNYdmbfseTzxNW
     userID: "taeuk"
     username: taeuk
   ```

   - `email` : dashdoard접속시 입력할 email

   - `hash` : dashdoard접속시 입력할 passward

     > [BCrypt Hash Generator](https://bcrypt-generator.com/) 에서 hash값을 생성할 수 있다.

   - `userID`, `username` : user정보

2. **add user information**

   ```
   $ kubectl -n auth edit cm dex
   ```

   >  vim editer로 변경

3. **rollout restart**

   dex manifast를 수정하고 난 후 해당 resource를 restart해주어야 한다.

   ```
   $ kubectl rollout restart deployment dex -n auth
   ```

4. **create namespace**

   이후 해당 ID/PW로 접속이 가능하지만, namespace가 지정되지 않아 자원 생성이 불가능하다. 

   이를 위해 namespace를 생성하자

   1. add profile

      ```
      $ vi profile.yaml
      ```

      ```yaml
      #profile.yaml
      apiVersion: kubeflow.org/v1beta1
      kind: Profile
      metadata:
        name: testuser
      spec:
        owner:
          kind: User
          name: winter4958@gmail.com
        resourceQuotaSpec: {}
      ```

      - `metadata.name` : kubeflow pipeline에서 사용할 namesapce의 name

      - `spec.owner`

        - `kind` : User로 고정
        - `name` : 위 dex resource에 추가한 User의 email

      - `resourceQuotaSpec` : 해당 namesapce의 resource 할당량 제한 (optional)

        ```
          resourceQuotaSpec:
            hard:
              cpu: "2"
              memory: 2Gi
              persistentvolumeclaims: "1"
              requests.nvidia.com/gpu: "1"
              requests.storage: "10Gi"
        ```

        - `cpu: "2"` : cpu제한 2개

        - `memory` : 메모리 제한 2 기가

        - `requests.nvidia.com/gpu` : 사용 가능항 GPU제한 1개

        - `persistentvolumeclaims` : volume 1개

        - `requests.storage` : 저장소 공간 제한 10GB

          > resourceQuotaSpec에 위 처럼 특정 값을 넣으면 아래의 에러가 발생
          >
          > ```
          > This step is in Error state with this message: task 'hibernation-project-9kj4p.set-config' errored: pods "hibernation-project-9kj4p-3036383870" is forbidden: failed quota: kf-resource-quota: must specify cpu,memory
          > ```

   2. apply

      ```
      $ kubectl apply -f profile.yaml
      ```

   3. edit

      profile 변경이 필요할 시

      ```
      $ kubectl edit profile <namespace_name>
      ```

      



## start

```
minikube start --driver=none \
  --kubernetes-version=v1.23.10  \
  --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key \
  --extra-config=apiserver.service-account-issuer=kubernetes.default.svc
```



### access dashboard

- port-forward 

  ```
  $ kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
  ```

  > ```
  > localhost:8080
  > ```

- port access

  ```
  $ minikube service list -n istio-system
  ```

  ```
  |--------------|-----------------------|-------------------|----------------------------|
  |  NAMESPACE   |         NAME          |    TARGET PORT    |            URL             |
  |--------------|-----------------------|-------------------|----------------------------|
  | istio-system | authservice           | No node port      |
  | istio-system | cluster-local-gateway | No node port      |
  | istio-system | istio-ingressgateway  | status-port/15021 | http://192.168.0.107:31478 |
  |              |                       | http2/80          | http://192.168.0.107:31355 |
  |              |                       | https/443         | http://192.168.0.107:30779 |
  |              |                       | tcp/31400         | http://192.168.0.107:31354 |
  |              |                       | tls/15443         | http://192.168.0.107:31375 |
  | istio-system | istiod                | No node port      |
  | istio-system | knative-local-gateway | No node port      |
  |--------------|-----------------------|-------------------|----------------------------|
  ```

  위 중 `istio-ingressgateway ` -`http2/80` 의 URL

  ```
  http://192.168.0.107:31355
  ```

  



### access from outside 

#### ngrok

**install** 

```
$ sudo snap install ngrok
```



**using**

> ```
> $  kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
> ```
>
> 위 명령어가 활성화 된 terminal이 열려있어야 함 

```
$ ngrok http 8080
```

```
ngrok by @inconshreveable                                                                                                                                           (Ctrl+C to quit)

Session Status                online
Session Expires               1 hour, 51 minutes
Version                       2.3.40
Region                        United States (us)
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://eb45-1-214-32-67.ngrok.io -> http://localhost:8090

Connections                   ttl     opn     rt1     rt5     p50     p90
                              2       0       0.00    0.00    300.58  300.92

HTTP Requests
-------------

GET /favicon.ico               302 Found
GET /                          302 Found
GET /                          302 Found
```

- `Session Status` : session의 상태. online일 경우 정상

- `Session Expires` : 남은 session의 만료 시간

  > 만료 시간이 지나면 다시 `./ngrok http {port}`명령어 입력해야함
  >
  > 만료 시간 없이 사용하려면 계정 연동. 방법은 아래에

- `Region` : ngrok agent가 ternal을 hoting하기 위한 region

- `Web Interface` : ngrok dashboard를 제공하는 URL

- `Forwarding` : ngrok에서 제공하는 ternal URL로, 이를 통해 외부에서도 local 한경에 접근할 수 있다. (http, https제공)





**account linking**

[공식 페이지](https://ngrok.com/) 에 로그인 후  `Your Authtoken`에서 token받고 아래 명령어

```
$ ngrok config add-authtoken {token값}
```



> remove
>
> ```
> $ sudo snap remove ngrok
> ```
>



#### External-IP

**istio-ingressgateway 의 spec.type이 `LoadBalancer` 임을 확인**

`nodeport`라 되어있으면 변경 필요

```
$ kubectl edit service -n istio-system istio-ingressgateway
```



1. **get `External-IP`** 

   아직은 `External-IP`이 `<pending> `일 것임

   ```
   $ kubectl get service -n istio-system istio-ingressgateway
   ```

   ```
   NAME                   TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                                      AGE
   istio-ingressgateway   LoadBalancer   10.110.207.155   <pending>     15021:31478/TCP,80:31355/TCP,443:30779/TCP,31400:31354/TCP,15443:31375/TCP   48m
   ```

   1. enable **MetalLB**

      minikube의 addons중 `metallb ` 확인

      ```
      $ minikube addons list
      ```

      ```
      | metallb                     | minikube | disabled     | 3rd party (MetalLB)            |
      ```

      

      enable `metallb`

      ```
      $ minikube addons enable metallb
      ```

      ```
      ❗  metallb is a 3rd party addon and not maintained or verified by minikube maintainers, enable at your own risk.
          ▪ Using image metallb/speaker:v0.9.6
          ▪ Using image metallb/controller:v0.9.6
      🌟  The 'metallb' addon is enabled
      ```

      

   2. set IP range

      ```
      $ minikube addons configure metallb
      ```

      ```
      -- Enter Load Balancer Start IP: 192.168.0.240
      -- Enter Load Balancer End IP: 192.168.0.249
          ▪ Using image metallb/speaker:v0.9.6
          ▪ Using image metallb/controller:v0.9.6
      ✅  metallb was successfully configured
      ```

      - `-- Enter Load Balancer Start IP` ,` -- Enter Load Balancer End IP` : **192.168.0.240** 부터 **192.168.0.249** 사이의 Host IP주소 할당

        > 너무 많이 할당하면 기존에 회사에서 사설 IP를 받아 쓰던 내부망 client들과 충돌이 있을 우려가 생기므로 조심
        
        ```
        -- Enter Load Balancer Start IP: 192.168.0.240
        -- Enter Load Balancer End IP: 192.168.0.249
        ```
        
        

      

   3. check `EXTERNAL-IP`

      ```
      $ kubectl get service -n istio-system istio-ingressgateway
      ```

      ```
      NAME                   TYPE           CLUSTER-IP       EXTERNAL-IP     PORT(S)                                                                      AGE
      istio-ingressgateway   LoadBalancer   10.110.207.155   192.168.0.240   15021:31478/TCP,80:31355/TCP,443:30779/TCP,31400:31354/TCP,15443:31375/TCP   54m
      ```

      `192.168.0.240` 의 외부 IP가 생겼음을 확인

   

2. **access from client**

   1. configure port poward

      iptime 공유기 사용 중이면 `192.168.0.1` 접속 후

      `관리도구 - 네이게이션의 고급 설정 - NAT/라우터 관리  - 포트포워드 설정`

      **새 규칙 추가**

      ```
      내부 IP : 192.168.0.107
      
      외부 포트 : 2222
      
      내부 포트 : 22
      ```

      > 외부 IP의 port 2222로 접속하면 내부 IP의 내부 port 22로 접속이 되게 설정

   2. access from client

      client terminal에서 

      ```
      ssh -L {외부 port}:{istio-ingressgatewa 의 EXTERNAL-IP}:80 {접속하고자 하는 서버의 ID}@{외부 IP} -p {외부 port}
      ```

      > 예시
      >
      > ```
      > ssh -L 2222:192.168.0.240:80 ainsoft@x.xxx.xx.xx -p 2222
      > ```

      이제 해당 terminal이 열려있는 상태에서

      browser에 `http://localhost:2222` 입력 시 kubeflow dashboard사용 사능

   





## uninstall



1. delete docker container

   ```
   $ docker rm -f $(docker ps -aq)
   ```

2. delete docker images

   ```
   $ docker rmi $(docker images -q)
   ```

3. delete minikube

   ```
   $ minikube stop
   $ minikube delete
   ```

   > minikube를 통해 실행되던 cluster를 삭제. minikube자체를 삭제한것이 아님.
   >
   > error
   >
   > - `env: ‘kubeadm’: No such file or directory`
   >
   >   이런 경우 minikube의 prifiles를 전부 삭제
   >
   >   ```
   >   $ sudo rm -rf ~/.kube ~/.minikube
   >   ```

4. delete kubectl 

   `kubectl`, `kubectl.sha256` 가 있는 위치에서 아래 명령어

   ```
   $ rm -rf kubectl
   $ rm -rf kubectl.sha256
   ```

   이후 아래 명령어

   ```
   $ sudo rm /usr/local/bin/kubectl
   ```

   


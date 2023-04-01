def with_kfp_client():
    from pipeline_utils import connet_client
    import json
    from pipeline_base_config import Config

    dashboard_cfg = {'user_n': 'winter4958@gmail.com', 'name_space': 'pipeline', 'host': 'http://localhost:8080', 'pw': '4958'}
    dashboard_cfg = Config(dashboard_cfg)
    client, session = connet_client(dashboard_cfg, return_session = True) 

    session_cookie = session.cookies.get_dict()
    sklear_iris_input = dict(instances = [
    [6.8, 2.8, 4.8, 1.4],
    [6.0, 3.4, 4.5, 1.6]
    ])

    HOST = "http://127.0.0.1:8081"

    headers = {'Host': "sklearn-iris-python2-predictor-default.pipeline.svc.cluster.local"}
    res = session.post(f"{HOST}/v1/models/sklearn-iris-python2:predict", 
                        headers=headers, 
                        cookies=session_cookie,
                        data=json.dumps(sklear_iris_input))

    print(f"res.json : {res.json()}")



# ---
        
def with_kserve_client():
    from kserve import (utils,
                        KServeClient)

    # 현재 실행중인 notebook의 namespace를 가져온다.
    # esle: default
    # namespace = utils.get_default_target_namespace()     

    namespace = "pipeline"


    service_name = "sklearn-iris-python2"
    kserve = KServeClient()     # istio ingressgateway가 연결되어있어야 한다

    # watch = True 를 주면 URL을 확인할 수 있지만 None을 return한다.
    # watch = False를 주면 inference service에 관한 dict을 return한다.
    isvc_resp = kserve.get(service_name, namespace = namespace)


    sklear_iris_input = dict(instances = [
        [6.8, 2.8, 4.8, 1.4],
        [6.0, 3.4, 4.5, 1.6]
    ])

    import requests
    import json

    isvc_url = isvc_resp['status']['address']['url']
    print(f"isvc_url ; {isvc_url}\n")


    # HTTPConnectionPool(host='sklearn-iris-test2.project-pipeline.svc.cluster.local', port=80): 
    # Max retries exceeded with url: /v1/models/sklearn-iris-test2:predict 
    # (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fe70aa8e9a0>: Failed to establish a new connection: [Errno -2] Name or service not known'))
    response = requests.post(isvc_url, json = json.dumps(sklear_iris_input))        
    print(response.text)




def tmp():
    sklear_iris_input = dict(instances = [
        [6.8, 2.8, 4.8, 1.4],
        [6.0, 3.4, 4.5, 1.6]
    ])


    import requests
    import kfp

    HOST = "http://127.0.0.1:8080/"     # https를 사용하면 안됨

    session = requests.Session()
    response = session.get(HOST)        # ssl접속 시 verify=False

    USERNAME = "winter4958@gmail.com"
    PASSWORD = "4958"



    headers = {
        "Content-Type" : "application/x-www-form-urlencoded",       # using 'form data'
    }

    data = {'login': USERNAME, "password": PASSWORD}
    session.post(response.url, headers = headers, data=data)
    session_cookie = session.cookies.get_dict()                     

    import json
    headers = {'Host': 'sklearn-iris-python2.pipeline.svc.cluster.local'}
    res = session.post(f"{HOST}v1/models/sklearn-iris-python2:predict", 
                        headers = headers,
                        cookies = session_cookie,
                        data = json.dumps(sklear_iris_input))
    print(res.json)

    # 또는 
    # curl -v -H "Host: sklearn-iris-python2.pipeline.svc.cluster.local" \
    # -d '{"instances": [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8]]}' \
    # -X POST http://sklearn-iris-python2.pipeline.svc.cluster.local:80/v1/models/sklearn-iris-python2:predict

    # <bound method Response.json of <Response [404]>>
    ### 경우 1.
    # kserve의 ingressgateway config맵이 잘못된 경로로 입력되어 배포되고 있다.
    # 이걸 수정
    # kubectl edit configmaps -n kserve inferenceservice-config             # 이건 미리 해놓자
    # ingress: |-
        # {
        #     "ingressGateway" : "knative-serving/knative-ingress-gateway",          >> 얘를  "kubeflow/kubeflow-gateway"
        #     "ingressService" : "istio-ingressgateway.istio-system.svc.cluster.local",
        #     "localGateway" : "knative-serving/knative-local-gateway",
        #     "localGatewayService" : "knative-local-gateway.istio-system.svc.cluster.local",
        #     "ingressDomain"  : "example.com",
        #     "ingressClassName" : "istio",
        #     "domainTemplate": "{{ .Name }}-{{ .Namespace }}.{{ .IngressDomain }}",
        #     "urlScheme": "http",
        #     "disableIstioVirtualHost": false
        # }
    ### 경우 2.  도메인 이름이 정확하지 않거나, 잘못 입력된 경우  >> 아님
    ### 경우 3.  DNS 서버가 제대로 동작하지 않는 경우
    ### 경우 2.  네트워크 연결이 안 된 경우 >> 아님


with_kfp_client()
# with_kserve_client()
# tmp()

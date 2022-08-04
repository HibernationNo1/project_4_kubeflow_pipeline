# AWS storage S3를 사용한 dataset관리

> 해당 기능은 access key노출 시 해커에 의한 과도한 금액 결제가 이루어질 수 있다는 위험성이 있어 중단하였습니다.



1. DVC를 활용하여 dataset을 S3에 저장 및 버전관리

2. AWS access key를 통해 S3에 접근, DVC pulI으로 dataset을 download

   **`arguments`** 

   - `--access_key_id`
   - `--secret_access_key`
   - `--bucket_name`

3. labelme code를 통해 개별 파일의 dataset을 학습이 가능한 단일 .json format 파일로 parsing

4. parsed dataset을 boto3(AWS python SDK)를 사용하여 S3에 Upload


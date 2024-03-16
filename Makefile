reinstall_package:
	@pip uninstall -y combat_iq || :
	@pip install -e .

build_container_local:
	docker build --tag=$$IMAGE:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 $$IMAGE:dev

# Step 1
allow_docker_push:
	gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

# Step 2
create_artifacts_repo:
	gcloud artifacts repositories create $$ARTIFACTSREPO --repository-format=docker --location=$$GCP_REGION --description="Repository for storing the docker container"

Step 3
build_for_production:
	docker build -t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

m2_build_image_production:
	docker build --platform linux/amd64 -t $$GCP_REGION-docker.pkg.dev/$$ARTIFACTSREPO/$$IMAGE:prod .

# m2_build_image_production:
# 	docker buildx build --push --platform linux/arm/v7,linux/arm64/v8,linux/amd64 -t $$GCP_REGION-docker.pkg.dev/$$ARTIFACTSREPO/$$IMAGE:prod .

# Step 4
push_image_production:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod

deploy_to_cloud_run:
	gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod --memory $$MEMORY --region $$GCP_REGION

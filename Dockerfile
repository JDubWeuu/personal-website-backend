FROM public.ecr.aws/lambda/python:3.12

# Set Lambda's working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy dependencies and install
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy main.py to the Lambda root (so CMD ["main.handler"] works)
COPY main.py ./

# Copy app folder contents to the Lambda root
COPY ./app ./app

# Define Lambda handler
CMD ["main.handler"]
FROM python:3.12-slim

RUN useradd -m agent
WORKDIR /home/agent

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY agent_protocol/ agent_protocol/
COPY web/ web/
COPY artifact_publisher.py main.py run_collab_long.py launch_hf_job.py manifest.json challenge_prompts.json ./

USER agent
ENV PYTHONUNBUFFERED=1

CMD ["python", "run_collab_long.py"]

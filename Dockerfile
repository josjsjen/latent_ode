FROM pytorch/pytorch

RUN pip install pandas sklearn matplotlib torchdiffeq

COPY . /home/jodiez/latent_ode/

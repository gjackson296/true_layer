FROM python:3.6

WORKDIR /code

RUN pip install pipenv

RUN pipenv install

COPY . .

EXPOSE 8888
CMD [ "jupyter-notebook", "--ip='*'", "--allow-root", "notebooks"]

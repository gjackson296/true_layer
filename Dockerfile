FROM python:3.6

WORKDIR /code

RUN pip install pipenv

COPY . .

RUN pipenv install

EXPOSE 8888
CMD ["pipenv", "run", "jupyter-notebook", "--ip='*'", "--allow-root"]

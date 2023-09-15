FROM apache/spark:3.4.0 as builder

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip


# Set the working directory
WORKDIR /app

# Copy the Spark project files to the container
COPY ./app /app

COPY ./requirements.txt /app

COPY ./app/Data /Data

# Install Python dependencies
RUN pip3 install -r requirements.txt
# RUN sed -i "s/localhost/$(curl http://checkip.amazonaws.com)/g" static/index.js

# Expose the port for the web application
EXPOSE 5432

# Set the entry point
CMD ["spark-submit", "server.py", "Data/movies.parquet", "Data/ratings.parquet"]
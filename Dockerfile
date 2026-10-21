FROM postgres:17-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git postgresql-server-dev-17 \
    libreadline-dev zlib1g-dev flex bison \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v0.8.1 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make OPTFLAGS="" \
    && make install \
    && rm -rf /tmp/pgvector

RUN git clone --branch release/PG17/1.7.0 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make install \
    && rm -rf /tmp/age

RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample

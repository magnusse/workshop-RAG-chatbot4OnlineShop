FROM ubuntu:latest
LABEL authors="sonke.magnussen"

ENTRYPOINT ["top", "-b"]
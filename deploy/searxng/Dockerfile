# Custom SearXNG image that bakes in our settings (JSON API enabled + science/wiki
# engines) so the research connector works on Cloud Run, where mounting a config
# volume isn't available. The base image's entrypoint substitutes SEARXNG_SECRET
# for the "ultrasecretkey" placeholder at startup.
FROM searxng/searxng:latest
COPY settings.yml /etc/searxng/settings.yml

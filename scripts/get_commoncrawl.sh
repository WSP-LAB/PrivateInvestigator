wget "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-18/segments/1712296815919.75/wet/CC-MAIN-20240412101354-20240412131354-00000.warc.wet.gz"
gzip -d CC-MAIN-20240412101354-20240412131354-00000.warc.wet.gz
mv CC-MAIN-20240412101354-20240412131354-00000.warc.wet ../dataset/commoncrawl.warc.wet

mkdir ../dataset/trec
wget https://tides.umiacs.umd.edu/webtrec/trecent/w3c-emails-part1.tar.gz
wget https://tides.umiacs.umd.edu/webtrec/trecent/w3c-emails-part2.tar.gz
wget https://tides.umiacs.umd.edu/webtrec/trecent/w3c-emails-part3.tar.gz
tar -zxf w3c-emails-part1.tar.gz -C ../dataset/trec
tar -zxf w3c-emails-part2.tar.gz -C ../dataset/trec
tar -zxf w3c-emails-part3.tar.gz -C ../dataset/trec
rm w3c-emails-part1.tar.gz
rm w3c-emails-part2.tar.gz
rm w3c-emails-part3.tar.gz

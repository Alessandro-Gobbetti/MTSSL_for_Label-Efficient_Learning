# script to download and unzip the UCI HAR dataset
# source: https://doi.org/10.24432/C54S4K
wget -O uci_har.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

# unzip the dataset
unzip -q uci_har.zip -d UCI_HAR
unzip -q UCI_HAR/UCI\ HAR\ Dataset.zip -d UCI_HAR

# clean up
rm uci_har.zip
rm UCI_HAR/UCI\ HAR\ Dataset.zip
[ -d "UCI_HAR/__MACOSX" ] && rm -rf UCI_HAR/__MACOSX
mv UCI_HAR/UCI\ HAR\ Dataset/* UCI_HAR/
rm -rf UCI_HAR/UCI\ HAR\ Dataset

# print message
echo "UCI HAR dataset downloaded and unzipped to UCI_HAR/"
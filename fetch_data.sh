#!/usr/bin/env bash
# Pull a handful of public UHC commercial medical policy PDFs into data/.
set -e
mkdir -p data
for p in bariatric-surgery cochlear-implants ambulance-services clinical-trials cognitive-rehabilitation; do
  curl -sL -A "Mozilla/5.0" -o "data/uhc-$p.pdf" "https://www.uhcprovider.com/content/dam/provider/docs/public/policies/comm-medical-drug/$p.pdf"
done
ls data

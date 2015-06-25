./layer7similarity.sh&
./layer6similarity.sh&
python layer5_pooling_similarity.py&
python layer5_similarity.py&
python layer4_similarity.py&
python layer3_similarity.py&
python layer2_pooling_norm_similarity.py&
python layer2_pooling_similarity.py&
python layer2_similarity.py&
python layer1_pooling_norm_similarity.py&
python layer1_pooling_similarity.py&
python layer1_similarity.py&

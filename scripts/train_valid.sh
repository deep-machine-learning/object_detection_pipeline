#!/bin/bash
cd $DARKHOME
# Configuration
DATA=cfg/voc.data
NET_CONFIG=cfg/myyolo-voc.cfg
START_WEIGHTS=weights/darknet19_448.conv.23
OUT_WEIGHTS=weights/yolo.weights
# AUGMENTED_DATA=cfg/augmented-voc.data
# python augmentation.py DATA AUGMENTED_DATA

if [ -n "${AUGMENTED_DATA}" ]; then
	# ./darknet detector train ${AUGMENTED_DATA} ${NET_CONFIG} ${WEIGHTS}
	./darknet detector valid ${AUGMENTED_DATA} ${NET_CONFIG} ${OUT_WEIGHTS}
else
	# ./darknet detector train ${DATA} ${NET_CONFIG} ${WEIGHTS}
	./darknet detector valid ${DATA} ${NET_CONFIG} ${OUT_WEIGHTS}
fi




./darknet detector valid ${DATA} ${NET_CONFIG} ${OUT_WEIGHTS}
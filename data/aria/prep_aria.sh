#!/bin/bash
set -e

SCENE=$1
ROOT=$2

if [ -z "$SCENE" ] || [ -z "$ROOT" ]; then
  echo "Usage: ./run.sh <scene> <root>"
  exit 1
fi

DATA_ROOT=$ROOT/datasets/$SCENE

python $ROOT/prep_aria_eq.py --vrs-file $DATA_ROOT/recording.vrs --mps-data-dir $DATA_ROOT/mps/slam/ --output-dir $ROOT/scannetpp_formatted/$SCENE

colmap model_converter --input_path $ROOT/scannetpp_formatted/$SCENE/colmap/ --output_path $ROOT/scannetpp_formatted/$SCENE/colmap/ --output_type TXT

rm $ROOT/scannetpp_formatted/$SCENE/colmap/*.bin

INPUT="$ROOT/scannetpp_formatted/$SCENE/colmap/cameras.txt"
OUTPUT="$ROOT/scannetpp_formatted/$SCENE/colmap/cameras_fish.txt"

> "$OUTPUT"

while IFS= read -r line; do
    if echo "$line" | grep -q '^#'; then
        echo "$line" >> "$OUTPUT"
    elif [ -z "$line" ]; then
        echo "" >> "$OUTPUT"
    else
        cam_id=$(echo "$line" | awk '{print $1}')
        model=$(echo "$line" | awk '{print $2}')
        width=$(echo "$line" | awk '{print $3}')
        height=$(echo "$line" | awk '{print $4}')
        fx=$(echo "$line" | awk '{print $5}')
        cx=$(echo "$line" | awk '{print $6}')
        cy=$(echo "$line" | awk '{print $7}')
        echo "$cam_id OPENCV_FISHEYE $width $height $fx $fx $cx $cx 0.0 0.0 0.0 0.0" >> "$OUTPUT"
    fi
done < "$INPUT"

echo "Converted to OPENCV_FISHEYE and saved as $OUTPUT"

cp $ROOT/scannetpp_formatted/$SCENE/colmap/cameras_fish.txt $ROOT/scannetpp_formatted/$SCENE/colmap/cameras.txt
cp $ROOT/scannetpp_formatted/$SCENE/colmap/cameras_fish.txt $ROOT/scannetpp_formatted/$SCENE/colmap/cameras_equidist.txt
# awk '{
#     if ($0 ~ /SIMPLE_PINHOLE/) {
#         gsub("SIMPLE_PINHOLE", "OPENCV");
#         print $0, "0 0 0 0 0";
#     } else {
#         print $0;
#     }
# }' scannetpp_formatted/$SCENE/colmap/cameras.txt > scannetpp_formatted/$SCENE/colmap/cameras_tmp.txt

# mv scannetpp_formatted/$SCENE/colmap/cameras_tmp.txt scannetpp_formatted/$SCENE/colmap/cameras.txt


INPUT="$ROOT/scannetpp_formatted/$SCENE/colmap/cameras_fish.txt"
OUTPUT="$ROOT/scannetpp_formatted/$SCENE/nerfstudio/transforms.json"
mkdir -p "$(dirname "$OUTPUT")"

while IFS= read -r line; do
    # 跳过注释或空行
    if echo "$line" | grep -q '^#'; then
        continue
    elif [ -z "$line" ]; then
        continue
    else
        set -- $line  # 用 $1, $2... 自动分字段

        cam_id=$1
        model=$2
        width=$3
        height=$4
        fx=$5
        fy=$6
        cx=$7
        cy=$8
        k1=$9
        k2=${10}
        k3=${11}
        k4=${12}

        # 输出为 JSON
        cat <<EOF > "$OUTPUT"
{
    "fl_x": $fx,
    "fl_y": $fy,
    "cx": $cx,
    "cy": $cy,
    "w": $width,
    "h": $height,
    "k1": $k1,
    "k2": $k2,
    "k3": $k3,
    "k4": $k4,
    "camera_model": "$model"
}
EOF
        echo "Wrote camera model to $OUTPUT"
        break  # 只取第一个相机
    fi
done < "$INPUT"

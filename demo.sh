SEQ_NAME=scene0050_00

echo [INFO] start mask clustering
python main.py --config demo --debug --seq_name_list $SEQ_NAME
echo [INFO] finish mask clustering

echo [INFO] visualizing
python -m visualize.vis_scene --config demo --seq_name $SEQ_NAME
echo [INFO] Please follow the instruction of pyviz to visualize the scene

# echo [INFO] start evaluating class agnostic masks
# python -m semantics.get_open-voc_features --config demo --seq_name_list $SEQ_NAME
# echo [INFO] finish evaluating class agnostic masks


# echo [INFO] start generating 3D masks
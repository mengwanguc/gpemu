

offloading_types=("tf" "tf-dsr-all" "tf-dslr-all" "dali" "ff")

for offloading_type in ${offloading_types[@]}; do
    echo "using $offloading_type"
    python eval_app_runner.py gan_ada_app.py /home/cc/data/ $offloading_type default_config.yaml
done
export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true

# CINN related settings
export FLAGS_use_cinn=true
export FLAGS_group_schedule_tiling_first=1
export FLAGS_enable_pir_api=1
export FLAGS_check_infer_symbolic=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_cinn_new_group_scheduler=1

# 是否打印 Program IR信息
export FLAGS_print_ir=true

# export GLOG_vmodule=ap_generic_drr_pass=6

# 调试信息
export GLOG_v=3
export FLAGS_enable_ap=1

export PYTHONPATH=$PYTHONPATH:/denghaodong/code/Paddle/build/python

export AP_WORKSPACE_DIR=/paddle-workspace/ap_workspace

# python /denghaodong/code/Paddle/ap-test/test_matmul_epilogue.py > log2.log 2>&1

python /denghaodong/code/Paddle/ap-test/test_matmul_layernorm.py> layernorm.log 2>&1
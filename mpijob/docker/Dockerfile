FROM uber/horovod:0.13.11-tf1.10.0-torch0.4.0-py3.5

RUN cd / && \
    git clone -b cnn_tf_v1.9_compatible https://github.com/tensorflow/benchmarks.git

CMD ["bash", "-c", "mpirun python /benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet101 --batch_size 64     --variable_update horovod --train_dir=/training_logs --summary_verbosity=3 --save_summaries_steps=10"]
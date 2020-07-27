import tensorflow as tf
import re
from datetime import datetime
import argparse


def log_train_metrics(metrics, step, fields=('sum', 'loss_ce', 'loss_bbox', 'loss_giou')):

    # step = metrics['
    for key, value in metrics.items():
        if key in fields:
            tf.summary.scalar(key, value, step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='')
    parser.add_argument('--output_dir', default='')

    args = parser.parse_args()

    logdir = ''
    if not args.output_dir:
        logdir = "/private/home/padentomasello/tb_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        logdir = args.output_dir

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    train_matcher = re.compile('((\d*: )?Epoch).*')
    valid_matcher = re.compile('((\d*: )? Average).*')

    with open(args.log_file, 'r') as infile:
        step = 0
        for line in infile:
            if (train_matcher.match(line)):
                parts = line.split('|')
                # print(line)
                metrics = {}
                for part in parts:
                    split = part.rsplit(':', 1);
                    key = split[0].strip();
                    # print(split)
                    value = float(split[1].strip());
                    metrics[key] = value
                log_train_metrics(metrics, step)
                step += 1
            elif(valid_matcher.match(line)):
                split = line.rsplit('=', 1)
                key = split[0];
                value = float(split[1])
                tf.summary.scalar(key, value, step)




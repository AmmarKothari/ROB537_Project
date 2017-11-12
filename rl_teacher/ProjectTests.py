import os
import os.path as osp
import random
from collections import deque
from time import time, sleep, strftime
import pdb
import multiprocess
import string
import csv
import copy
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel_trpo

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef
from rl_teacher.video import SegmentVideoRecorder

from rl_teacher.teach import TraditionalRLRewardPredictor, ComparisonRewardPredictor
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from rl_teacher.video import write_segment_to_video, load_path


CLIP_LENGTH = 1.5


class ProjectTests(object):
	def __init__(self):
		self.test_time = int(10e3)
		self.test_time = int(30)
		self.num_workers = 1
		# self.num_workers = int(multiprocess.cpu_count()/2)

	def allTests(self):
		r1 = self.ToyRLRecordData() #consider this a baseline
		print('Testing Complete: %s' %r1)
		r2 = self.ToySyntheticRecordData()
		print('Testing Complete: %s' %r2)


	def ToyRLRecordData(self):
		test_name = 'ToyRLRecordData'
		env_id = 'reacher'
		run_name = "%s/%s-%s" % (env_id, test_name, int(time()))
		summary_writer = make_summary_writer(run_name)
		env = make_with_torque_removed(env_id)
		num_timesteps = int(self.test_time) # sets the time the iterations run for, but not human time!
		experiment_name = slugify(test_name)
		seed = 1
		max_kl = 0.001
		worker_count = self.num_workers

		# Traditional RL Reward Predictor
		predictor = TraditionalRLRewardPredictor(summary_writer)

		# record videos every once in a while
		predictor = SegmentVideoRecorder(predictor,
										env, 
										save_dir=osp.join('/tmp/rl_teacher_vids', run_name),
										save_video = True,
										save_paths = True,
										checkpoint_interval=500)
		# worker_count = 1
		policy = train_parallel_trpo(
						env_id=env_id,
						make_env=make_with_torque_removed,
						predictor=predictor,
						summary_writer=summary_writer,
						workers=worker_count,
						runtime=(num_timesteps),
						max_timesteps_per_episode=get_timesteps_per_episode(env),
						timesteps_per_batch=8000,
						max_kl=max_kl,
						seed=seed,
					)
		fn = self.saveModel(run_name, predictor, policy)
		return run_name

	def ToySyntheticRecordData(self):
		test_name = 'ToySyntheticRecordData'
		env_id = 'reacher'
		run_name = "%s/%s-%s" % (env_id, test_name, int(time()))
		summary_writer = make_summary_writer(run_name)
		env = make_with_torque_removed(env_id)
		num_timesteps = int(self.test_time) # sets the time the iterations run for, but not human time!
		experiment_name = slugify(test_name)

		agent_logger = AgentLogger(summary_writer)
		pretrain_labels = 100
		n_labels = 1000 # feedback provided by the supervisor
		pretrain_iters = 100
		# pretrain_iters = 10000


		label_schedule = LabelAnnealer(
				agent_logger,
				final_timesteps=num_timesteps,
				final_labels=n_labels,
				pretrain_labels=pretrain_labels)
		comparison_collector = SyntheticComparisonCollector() # synthetic collection

		# comparison reward estimator
		predictor = ComparisonRewardPredictor(
			env,
			summary_writer,
			comparison_collector=comparison_collector,
			agent_logger=agent_logger,
			label_schedule=label_schedule,
		)


		print("Starting random rollouts to generate pretraining segments. No learning will take place...")
		worker_count = int(multiprocess.cpu_count()/2)
		pretrain_segments = segments_from_rand_rollout(
			env_id,
			make_with_torque_removed,
			n_desired_segments=pretrain_labels * 2,
			clip_length_in_seconds=CLIP_LENGTH,
			workers=worker_count)
		for i in range(pretrain_labels):  # Turn our random segments into comparisons
			comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])



		# Start the actual training
		for i in range(pretrain_iters):
			predictor.train_predictor()  # Train on pretraining labels
			if i % 100 == 0:
				print("%s/%s predictor pretraining iters... " % (i, pretrain_iters))
		
		# synthetically labelling data		
		while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
			comparison_collector.label_unlabeled_comparisons()

		# record videos every once in a while
		predictor = SegmentVideoRecorder(predictor,
										env, 
										save_dir=osp.join('/tmp/rl_teacher_vids', run_name),
										save_video = True,
										save_paths = True,
										checkpoint_interval=500)
		seed = 1
		max_kl = 0.001
		policy = train_parallel_trpo(
						env_id=env_id,
						make_env=make_with_torque_removed,
						predictor=predictor,
						summary_writer=summary_writer,
						workers=worker_count,
						runtime=(num_timesteps),
						max_timesteps_per_episode=get_timesteps_per_episode(env),
						timesteps_per_batch=8000,
						max_kl=max_kl,
						seed=seed,
					)
		fn = self.saveModel(run_name, predictor, policy)
		return run_name

	def TestID(self):
		fn = 'TestID.csv'
		try:
			with open(fn, 'r', newline='') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					pass # don't care just want last value
			TestID = int(row[0]) + 1
		except:
			TestID = 0
		timestr = strftime("%Y%m%d-%H%M%S")
		with open(fn, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([TestID, timestr])
		return TestID

	def HumanSyntheticRecordData(self):
		# unique_ID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
		unique_ID = self.TestID()
		test_name = 'ToyHumanRecordData' + '_' + str(unique_ID)
		env_id = 'reacher'
		run_name = "%s/%s-%s" % (env_id, test_name, int(time()))
		summary_writer = make_summary_writer(run_name)
		env = make_with_torque_removed(env_id)
		num_timesteps = int(self.test_time) # sets the time the iterations run for, but not human time!
		experiment_name = slugify(test_name)

		agent_logger = AgentLogger(summary_writer)
		pretrain_labels = 2
		n_labels = 100 # feedback provided by the supervisor
		# pretrain_iters = 10000
		pretrain_iters = 100
		worker_count = self.num_workers


		label_schedule = LabelAnnealer(
				agent_logger,
				final_timesteps=num_timesteps,
				final_labels=n_labels,
				pretrain_labels=pretrain_labels)

		# comparison reward estimator
		bucket = 'gs://rl-teacher-kotharia'
		os.environ['RL_TEACHER_GCS_BUCKET'] = bucket
		assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
		comparison_collector = HumanComparisonCollector(env_id, experiment_name=experiment_name)

		predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
        )

		print("Starting random rollouts to generate pretraining segments. No learning will take place...")
		pretrain_segments = segments_from_rand_rollout(
			env_id,
			make_with_torque_removed,
			n_desired_segments=pretrain_labels * 2,
			clip_length_in_seconds=CLIP_LENGTH,
			workers=worker_count)
		for i in range(pretrain_labels):  # Turn our random segments into comparisons
			comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])
		
		# Sleep until the human has labeled most of the pretraining comparisons
		while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
			comparison_collector.label_unlabeled_comparisons()	
			print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (
				len(comparison_collector.labeled_comparisons), pretrain_labels))
			sleep(5)
		
		# Start the actual training
		for i in range(pretrain_iters):
			predictor.train_predictor()  # Train on pretraining labels
			if i % 100 == 0:
				print("%s/%s predictor pretraining iters... " % (i, pretrain_iters))
				
		# record videos every once in a while
		predictor = SegmentVideoRecorder(predictor,
										env, 
										save_dir=osp.join('/tmp/rl_teacher_vids', run_name),
										save_video = False,
										save_paths = True,
										checkpoint_interval=500)
		seed = 1
		max_kl = 0.001
		policy = train_parallel_trpo(
						env_id=env_id,
						make_env=make_with_torque_removed,
						predictor=predictor,
						summary_writer=summary_writer,
						workers=worker_count,
						runtime=(num_timesteps),
						max_timesteps_per_episode=get_timesteps_per_episode(env),
						timesteps_per_batch=8000,
						max_kl=max_kl,
						seed=seed,
					)
		self.recordTrajectoriesVideos(run_name, env)
		fn = self.saveModel(run_name, predictor, policy)
		s = ''
		with open(run_name + '_stats.txt', 'w') as f:
			s += 'Environment: %s \n' %env_id
			s += 'Time Steps: %s \n' %num_timesteps 
			s += 'Pretrain Labels: %s \n' %pretrain_labels
			s += 'Example Labels: %s \n' %n_labels
			s += 'Pretrain Iterations: %s \n' %pretrain_iters
			f.write(s)
		return run_name

	def recordTrajectoriesVideos(self, run_name, env):
		# seperate into two functions: load and record
		for file in os.listdir(run_name):
			if file.endswith('csv'):
				file = os.path.join(run_name, file)
				#load values
				path = load_path(file)
				write_segment_to_video(path, file + '.mp4', env)


	def readTensorBoard(self, fn):
		# fn = '/home/kotharia/tb/rl-teacher/ShortHopper-v1/syn-1400-1510303936'
		ea = event_accumulator.EventAccumulator(fn,
				size_guidance={ # see below regarding this argument
				event_accumulator.COMPRESSED_HISTOGRAMS: 500,
				event_accumulator.IMAGES: 4,
				event_accumulator.AUDIO: 4,
				event_accumulator.SCALARS: 0,
				event_accumulator.HISTOGRAMS: 1,
				})
		ea.Reload()
		return ea

	def saveTensorBoardToCSV(self, fn):
		board = self.readTensorBoard(fn)
		tags = board.Tags()
		holder = None
		for t in tags['scalars']:
			vals = np.array([v.value for v in board.Scalars(t)]).reshape(-1,1)
			save_fn = os.path.join(fn, t) + '.csv'
			os.makedirs(osp.dirname(save_fn), exist_ok=True)
			with open(save_fn, 'w', newline="") as csvfile:
				writer = csv.writer(csvfile)
				for v in vals:
					writer.writerow(v)
		print('Files Saved to %s' %fn)
		pdb.set_trace()
		# holder = np.array(holder).transpose()
		


	def saveModel(self, run_name, predictor, policy):
		# pdb.set_trace()
		# make new folders for saving data
		policy_save_dir = osp.join(osp.join('/tmp/rl_teacher_vids', run_name), 'policy')
		cost_save_dir = osp.join(osp.join('/tmp/rl_teacher_vids', run_name), 'cost')
		os.makedirs(policy_save_dir, exist_ok=True)
		os.makedirs(cost_save_dir, exist_ok=True)
		saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		predictor.predictor.sess.run(init_op)
		cost_func_save_path = saver.save(predictor.predictor.sess, cost_save_dir + '/cost')
		print('Cost Function Saved: %s' %cost_func_save_path)
		policy_save_path = saver.save(policy.session, policy_save_dir + '/policy')
		print('Policy Saved: %s' %policy_save_path)
		return cost_func_save_path

	def loadModel(self, fn):
		pdb.set_trace()
		loader = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)
			loader = tf.train.import_meta_graph(fn+'.meta')
			loader.restore(sess, tf.train.latest_checkpoint(fn.strip('cost')))
			print('Model Restored')
			# would need to reinitalize all the other stuff and assign the session to the correct location,
			# but it seems like it is working for now

	def testRecordTraj(self):
		base_dir = '/tmp/rl_teacher_vids/reacher/'
		run_name = 'ToyHumanRecordData_7-1510430775'
		env_id = 'reacher'
		env = make_with_torque_removed(env_id)
		PT.recordTrajectoriesVideos(base_dir + run_name, env)

	# def plto




def main():
	PT = ProjectTests()
	fn = '/home/kotharia/tb/rl-teacher/reacher/ToyRLRecordData-1510371722'
	PT.saveTensorBoardToCSV(fn)
	# PT.ToyRLRecordData()
	# cf_fn = PT.ToySyntheticRecordData()
	# PT.loadModel(cf_fn)
	# PT.HumanSyntheticRecordData()




if __name__ == '__main__':
	main()
#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run :
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
import tf
from std_srvs.srv import Empty
import json
import requests

# Joe Lisk code
#from chatGPTService import getTargetFromGPT
#from jointGPTPrompt import system_string, user_string
# import error: moduleNotFound

sceneObjects = {
  "can" : {
    "go_to": {
      "pre-grasp": {
        "position": {"x": -0.008528227056494208, "y": -0.4545102051969075, "z": 0.031031940091151258},
        "orientation": {"x": 0.05375719323178181, "y": -0.7757844834263641, "z": 0.6283301669563721, "w": 0.021674887388595025}
      },
      "grasp": {
        "position": {"x": 0.002531766299089388, "y": -0.6028243179191973, "z": 0.02571023473658379},
        "orientation": {"x": 0.046498299219275584, "y": -0.7693984296865566, "z": 0.6370337922518455, "w": 0.0072050048444401256}
      }
    }
  },
  "trash_bin": {
    "go_to": {
      "position": {"x": -0.007253318946339758, "y": 0.4578600037286752, "z": 0.1964750840769468},
      "orientation": {"x": 0.08187678131496287, "y": 0.6676512236268767, "z": 0.7399520009621545, "w": 0.003012066257614919}
    }
  },
  "gripper": {
    "open": 1,
    "close": 0.25
  },
  "home": {
    "go_to": "home" # named position in Kinova
  }
}


cachedSolution = {
  "prompts": [
    "pick up the trash",
    "take out the trash",
    "remove the trash",
    "take out the garbage"
  ],
  "actions": [
    'sceneObjects["gripper"]["open"]',
    'sceneObjects["can"]["go_to"]',
    'sceneObjects["gripper"]["close"]',
    'sceneObjects["trash_bin"]["go_to"]',
    'sceneObjects["gripper"]["open"]',
    'sceneObjects["home"]["go_to"]'
  ]
}

# NOTE
# calling HuggingFace service
def getSentenceSimilarityFromHF(sentences):
  #import json
  #import requests
  api_token = ""

  API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
  headers = {"Authorization": f"Bearer {api_token}"}

  def query(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.json()

  data = query(
    {
        "inputs": {
            "source_sentence": "pick up the trash",
            "sentences":[sentences]
        }
    })

  return data

## [0.605, 0.894]


# from chatGPTService
import os
import openai
import rospy

openai.api_key = os.environ["OPENAI_API_KEY"]

def getTargetFromGPT(system_string, user_string):
    messages = [{"role": "system", "content": system_string}]
    messages.append({"role": "user", "content": user_string})
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages)
    response = completion['choices'][0]['message']['content']
    rospy.loginfo('Successfully got response from GPT-4!')
    rospy.loginfo('Response from GPT-4:')
    rospy.loginfo(response)
    target = eval(response)
    return target

# From jointGPTPrompt
system_string = """You are a Kinova gen3 arm that is tasked with determining which objects in the environment are trash, picking up the
            objects that are trash, and moving them to a specified point. You have 7 degrees of freedom and a gripper. It is possible
            that none of the objects are trash. You have a known starting cartesian pose called "home".

            Your output should be a string that represents the object that is trash (if any).

            Also, I'm going to execute your code in Python, so please format any steps you outline as a comment or the code will
            fail to run.
            """

# user_string = """There are two objects in the evnironment. beer_A: an empty can of beer at the position (x=0.7, y=0.0, z=0.0). And beer_B: an full, unopened can of beer at (x=0, y=-0.7, z=0).
#                Provide the variable name of the object that is trash. Also, I'm going to execute your code in Python, so please do not explain, just provide the variable name"""

#Joe Lisk code
#Helper function for joint angles
# def deg2rad(x):
#   return x * pi / 180

#Adapted from https://github.com/turtlebot/turtlebot/blob/kinetic/turtlebot_teleop/scripts/turtlebot_teleop_key
msg = """
Give your environment for your Kinova arm!
---------------------------
Describe your workspace and the objects present.

ChatGPT will then help control the robot.

Currently, you need to supply the poses

Of the objects in the form:

position.x = num, position.y = num, position.z = num, orientation.x = num, orientation.y = num, orientation.z = num, orientation.w = num

Grasp pose detection coming soon!

commands
---------------------------
enter : sends your input to ChatGPT

CTRL-C to quit
"""

def getUserString():
    str = input()
    return str

class ExampleMoveItTrajectories(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(ExampleMoveItTrajectories, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_it_service')

    try:
      self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
      if self.is_gripper_present:
        gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
        self.gripper_joint_name = gripper_joint_names[0]
      else:
        self.gripper_joint_name = ""
      self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

      # Create the MoveItInterface necessary objects
      arm_group_name = "arm"
      self.robot = moveit_commander.RobotCommander("robot_description")
      self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
      self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
      self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True


  def reach_named_position(self, target):
    arm_group = self.arm_group

    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(trajectory_message, wait=True)

  def get_cartesian_pose(self):
    arm_group = self.arm_group

    # Get the current pose and display it
    pose = arm_group.get_current_pose()
    rospy.loginfo("Actual cartesian pose is : ")
    rospy.loginfo(pose.pose)

    return pose.pose

  def reach_cartesian_pose(self, target_obj, target_pose, tolerance, constraints):
    arm_group = self.arm_group

    # Set the tolerance
    arm_group.set_goal_position_tolerance(tolerance)

    # Set the trajectory constraint if one is specified
    if constraints is not None:
      arm_group.set_path_constraints(constraints)

    # Joe Lisk code
    # Just spotting ChatGPT the trash can pose
    if target_obj == 'trash_can':
      target_pose.position.x = -0.007253318946339758
      target_pose.position.y = 0.4578600037286752
      target_pose.position.z = 0.1964750840769468
      target_pose.orientation.x = 0.08187678131496287
      target_pose.orientation.y = 0.6676512236268767
      target_pose.orientation.z = 0.7399520009621545
      target_pose.orientation.w = 0.003012066257614919

    # Get the current Cartesian Position
    arm_group.set_pose_target(target_pose)

    # Plan and execute
    rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)

  def reach_gripper_position(self, relative_position):
    gripper_group = self.gripper_group
 
    # We only have to move this joint because all others are mimic!
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()

    try:
      val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
      return val
    except:
      return False

def main():
  example = ExampleMoveItTrajectories()

  # For testing purposes
  success = example.is_init_success
  try:
      rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
  except:
      pass


  # comment out when using Gazebo sim. for real arm need to send home first
  if success:
    rospy.loginfo("Reaching Named Target Home...")
    success &= example.reach_named_position("home")
    print(success)
  
  # Joe Lisk code
  # Prompt 1: deciding which of the detected objects is trash
  # print(msg)
  # gettingInput = True
  # while(gettingInput):
  #   usr_str = getUserString()
  #   if usr_str != '':
  #     print("THIS IS YOUR INPUT:")
  #     print(usr_str)
  #     user_string = usr_str
  #     gettingInput = False

  # # Joe Lisk code
  # # chatgpt service
  # gpt_s = getTargetFromGPT(system_string, user_string)
  # print(str(gpt_s))

  # Human-robot interaction test
  # if gpt_s["certainty"] == "unsure":
  #   print(msg)
  #   gettingInput = True
  #   while(gettingInput):
  #     usr_str = getUserString()
  #     if usr_str != '':
  #       print("THIS IS YOUR INPUT:")
  #       print(usr_str)
  #       user_string = usr_str
  #       gettingInput = False

  #   # Joe Lisk code
  #   # chatgpt service
  #   gpt_s = getTargetFromGPT(system_string, user_string)


  # Prompt 2: grasping
  print(msg)
  gettingInput = True
  while(gettingInput):
    usr_str = getUserString()
    if usr_str != '':
      print("THIS IS YOUR INPUT:")
      print(usr_str)
      user_string = usr_str
      gettingInput = False

  # Joe Lisk code
  # chatgpt service
  gpt_list = []
  promptCheck = user_string.split(". Example")[0]
  cacheThreshold = 0.3
  if promptCheck not in cachedSolution["prompts"] and getSentenceSimilarityFromHF(promptCheck)[0] < cacheThreshold:
    gpt_list = getTargetFromGPT(system_string, user_string)
  elif getSentenceSimilarityFromHF(promptCheck)[0] == 1.0: # don't bother with dialouge for exact match
    gpt_list = cachedSolution["actions"]
  else:
    print(f"Found similar prompt. Did you mean: {cachedSolution['prompts']}?")
    gettingInput = True
    while(gettingInput):
      usr_str = getUserString()
      if usr_str != '':
        print("THIS IS YOUR INPUT:")
        print(usr_str)
        #user_string = usr_str
        gettingInput = False
      
    if usr_str == "Yes" or usr_str == "yes":
      gpt_list = cachedSolution["actions"]
    else:
      gpt_list = getTargetFromGPT(system_string, user_string)


  #targetObj = gpt_list[0] # string
  #targetPreGraspPose = gpt_list[1] # dictionary
  #targetGraspPose = gpt_list[2] # dictionary
  # targetObj = 'beer_A'
  # targetPreGraspPose = gpt_list[0] # dictionary
  # targetGraspPose = gpt_list[1] # dictionary

  for a in range(len(gpt_list)):

    # for the unknown objects
    if gpt_list[a] == "unknown": # .includes
      # call gptServiceAgain()
      break

    action = eval(gpt_list[a])
    if success:
      rospy.loginfo("Performing action...")
      actual_pose = example.get_cartesian_pose()

      if action == 0.25 or action == 1:
        rospy.loginfo("do gripper stuff...")
        success &= example.reach_gripper_position(action)
        print (success)

      elif action == "home":
        rospy.loginfo("do home stuff...")
        success &= example.reach_named_position("home")
        print (success)

      elif "pre-grasp" in action.keys():
        rospy.loginfo("do object grabbing stuff...")
        targetPreGraspPose = action['pre-grasp']
        actual_pose.position.x = targetPreGraspPose['position']['x']
        actual_pose.position.y = targetPreGraspPose['position']['y']
        actual_pose.position.z = targetPreGraspPose['position']['z']
        actual_pose.orientation.x = targetPreGraspPose['orientation']['x']
        actual_pose.orientation.y = targetPreGraspPose['orientation']['y']
        actual_pose.orientation.z = targetPreGraspPose['orientation']['z']
        actual_pose.orientation.w = targetPreGraspPose['orientation']['w']

        targetObj = 'beer_B'
        success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
        print(success)

        if success:
          actual_pose = example.get_cartesian_pose()
          targetGraspPose = action['grasp']
          actual_pose.position.x = targetGraspPose['position']['x']
          actual_pose.position.y = targetGraspPose['position']['y']
          actual_pose.position.z = targetGraspPose['position']['z']
          actual_pose.orientation.x = targetGraspPose['orientation']['x']
          actual_pose.orientation.y = targetGraspPose['orientation']['y']
          actual_pose.orientation.z = targetGraspPose['orientation']['z']
          actual_pose.orientation.w = targetGraspPose['orientation']['w']

          targetObj = 'beer_B'
          success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
          print(success)

      elif "position" in action.keys():
        rospy.loginfo("do trash can stuff...")
        targetPose = action
        actual_pose.position.x = targetPose['position']['x']
        actual_pose.position.y = targetPose['position']['y']
        actual_pose.position.z = targetPose['position']['z']
        actual_pose.orientation.x = targetPose['orientation']['x']
        actual_pose.orientation.y = targetPose['orientation']['y']
        actual_pose.orientation.z = targetPose['orientation']['z']
        actual_pose.orientation.w = targetPose['orientation']['w']

        targetObj = 'beer_B'
        success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
        print(success)

  # # Pre-Grasp
  # if success:
  #   rospy.loginfo("Reaching Cartesian Pose...")

  #   actual_pose = example.get_cartesian_pose()

  #   actual_pose.position.x = targetPreGraspPose['position']['x']
  #   actual_pose.position.y = targetPreGraspPose['position']['y']
  #   actual_pose.position.z = targetPreGraspPose['position']['z']
  #   actual_pose.orientation.x = targetPreGraspPose['orientation']['x']
  #   actual_pose.orientation.y = targetPreGraspPose['orientation']['y']
  #   actual_pose.orientation.z = targetPreGraspPose['orientation']['z']
  #   actual_pose.orientation.w = targetPreGraspPose['orientation']['w']

  #   success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
  #   print (success)

  # if success:
  #   actual_pose = example.get_cartesian_pose()
  #   rospy.loginfo("Cartesian Pose: PRE-GRASP")
  #   rospy.loginfo(actual_pose)

  # # Grasp
  # if success:
  #   rospy.loginfo("Reaching Cartesian Pose...")

  #   actual_pose = example.get_cartesian_pose()

  #   actual_pose.position.x = targetGraspPose['position']['x']
  #   actual_pose.position.y = targetGraspPose['position']['y']
  #   actual_pose.position.z = targetGraspPose['position']['z']
  #   actual_pose.orientation.x = targetGraspPose['orientation']['x']
  #   actual_pose.orientation.y = targetGraspPose['orientation']['y']
  #   actual_pose.orientation.z = targetGraspPose['orientation']['z']
  #   actual_pose.orientation.w = targetGraspPose['orientation']['w']
  
  #   success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
  #   print (success)

  # if success:
  #   actual_pose = example.get_cartesian_pose()
  #   rospy.loginfo("Cartesian Pose: GRASP")
  #   rospy.loginfo(actual_pose)

  #   # rospy.loginfo("Closing the gripper 100%...")
  #   rospy.loginfo("Closing the gripper 75%...")
  #   success &= example.reach_gripper_position(0.25)
  #   print (success)

  # if success:
  #   rospy.loginfo("Reaching Cartesian Pose...")

  #   #TODO: hack for now
  #   # actual_pose = example.get_cartesian_pose()

  #   success &= example.reach_cartesian_pose('trash_can', target_pose=actual_pose, tolerance=0.01, constraints=None)
  #   print (success)

  # if success:
  #   actual_pose = example.get_cartesian_pose()
  #   rospy.loginfo("Cartesian Pose: TRASH CAN")
  #   rospy.loginfo(actual_pose)

  # if example.is_gripper_present and success:
  #   rospy.loginfo("Opening the gripper...")
  #   success &= example.reach_gripper_position(1)
  #   print (success)

  # if success:
  #   rospy.loginfo("Reaching Named Target Home...")
  #   success &= example.reach_named_position("home")
  #   print (success)

  # comment out when using Gazebo sim and running real arm tests. this is just to guide the robot to a safe off position when done testing.
  # if success:
  #   rospy.loginfo("Reaching Target Off Position...")

  #   off_list = ["off_position",
  #     {"position": {"x": 0.002531766299089388, "y": -0.6028243179191973, "z": 0.02571023473658379},
  #       "orientation": {"x": 0.046498299219275584, "y": -0.7693984296865566, "z": 0.6370337922518455, "w": 0.0072050048444401256}}]

  #   targetGraspPose = off_list[1] # dictionary

  #   actual_pose = example.get_cartesian_pose()

  #   actual_pose.position.x = targetGraspPose['position']['x']
  #   actual_pose.position.y = targetGraspPose['position']['y']
  #   actual_pose.position.z = targetGraspPose['position']['z']
  #   actual_pose.orientation.x = targetGraspPose['orientation']['x']
  #   actual_pose.orientation.y = targetGraspPose['orientation']['y']
  #   actual_pose.orientation.z = targetGraspPose['orientation']['z']
  #   actual_pose.orientation.w = targetGraspPose['orientation']['w']
  
  #   targetObj = 'off'
  #   success &= example.reach_cartesian_pose(targetObj, target_pose=actual_pose, tolerance=0.01, constraints=None)
  #   print(success)

  # For testing purposes
  rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

  if not success:
      rospy.logerr("The example encountered an error.")

if __name__ == '__main__':
  main()


# Prompt 1:
# Your object detection system found the following objects in the scene: 1 person, 1 bottle, 2 bowls, 1 sandwich, 2 chairs, 2 dining tables. Which one of these objects is most likely to be trash? Please only put quotation marks around the beginning and end of your response. 

# Modified to demo uncertainty: (it always chooses bottle)
# Your object detection system found the following objects in the scene: 1 person, 1 bottle, 2 bowls, 1 sandwich, 2 chairs, 2 dining tables. Which one of these objects is most likely to be trash? If you are unsure or need more context, you can ask a question about the most likely object. Please only put quotation marks around the beginning and end of your response. 

# Modified to guarantee uncertainty
# Your object detection system found the following objects in the scene: 2 cans. Between the two cans, canA and canB, which one of these objects is most likely to be trash? If you are unsure or need more context, you can ask a question about the most likely object. Please only put quotation marks around the beginning and end of your response. Please format your response in the following way: {"most_likely_object": "object", "certainty": "sure_or_unsure", "explanation": "your_explanation"}. Only provide the dictionary.

# Human Response:
# Can A is full, Can B is empty. Which one is most likely to be trash? Please format your response in the following way: {"most_likely_object": "object", "certainty": "sure_or_unsure", "explanation": "your_explanation"}. Only provide the dictionary.

# Prompt if unknown
# The _ object is empty. Is it trash?

# number of times, prompting GPT (x = failure, o = success, c = crash):
# prompt 1: c, o, o, o, o, o
# prompt 2: c, c(it gave me the correct list, but it crashed for some reason), c (was correct, crashed), o

# Prompt 2:
# user input: The bottle has a pre-grasp pose of target_pose.position.x = 0.5338950844247246, target_pose.position.y = 0.013882795317536337, target_pose.position.z = 0.03560667199606036, target_pose.orientation.x = 0.5233042320042509, target_pose.orientation.y = 0.4906332994586916, target_pose.orientation.z = 0.47538816211295915, target_pose.orientation.w = 0.509350313194742 and a grasp pose of, target_pose.position.x = 0.7069753526986098, target_pose.position.y = 0.012251526063553923, target_pose.position.z = 0.030152609462021807, target_pose.orientation.x = 0.517385163838614, target_pose.orientation.y = 0.49491109344252465, target_pose.orientation.z = 0.4797642455445715, target_pose.orientation.w = 0.507150737477788. Provide a Python list containing 2 elements: the pre-grasp pose of the bottle as a dictionary, the grasp pose of the bottle as a dictionary. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable.

# Prompt NEW:
# iteration: pick up the trash. Example output: [sceneObjects["gripper"]["open"], sceneObjects["bottle"]["go_to"], sceneObjects["gripper"]["close"], sceneObjects["trash"]["go_to"], sceneObjects["gripper"]["open"], sceneObjects["home"]["go_to"]]. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable. The trash object is a can.
# new iteration - to break the cache:
# Now, provide code to pick up the trash. Example output: ['sceneObjects["gripper"]["open"]', 'sceneObjects["trash_object"]["go_to"]', 'sceneObjects["gripper"]["close"]', 'sceneObjects["trash_bin"]["go_to"]', 'sceneObjects["gripper"]["open"]', 'sceneObjects["home"]["go_to"]']. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable. You correctly identified the bottle as the trash object, but it actually a can.

# Should trigger the cache
# pick up trash. Example output: [sceneObjects["gripper"]["open"], sceneObjects["bottle"]["go_to"], sceneObjects["gripper"]["close"], sceneObjects["trash"]["go_to"], sceneObjects["gripper"]["open"], sceneObjects["home"]["go_to"]]. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable. The trash object is a can.

# Example:  you could do a string split at "Example" "pick up the trash"





##################
# RESPONSE FROM GPT-4
#Your request seems to ask for two different things: a string that represents the object that is trash, and a list of steps for the robot arm to pick up and dispose of the trash. Please specify which one you want, or if you want both.
#Adding the Now, provide code to

###################
# RESPONSE FROM GPT-4
# [sceneObjects["gripper"]["open"], 
# sceneObjects["can"]["go_to"], 
# sceneObjects["gripper"]["close"], 
# sceneObjects["trash_bin"]["go_to"], 
# sceneObjects["gripper"]["open"], 
# sceneObjects["home"]["go_to"]]

#########################
#Response from GPT-4:
#[INFO] [1731979573.509680, 603.479000]: 
# ['sceneObjects["gripper"]["open"]', 
# 'sceneObjects["can"]["go_to"]', 
# 'sceneObjects["gripper"]["close"]', 
# 'sceneObjects["trash_bin"]["go_to"]', 
# 'sceneObjects["gripper"]["open"]', 
# 'sceneObjects["home"]["go_to"]']

#############################
# joe@joe-vm:~/catkin_ws/src/ros_kortex/kortex_examples/src/move_it$ rosrun kortex_examples example_move_it_trajectories.py __ns:=my_gen3_lite
# [ INFO] [1731979805.249013245]: Loading robot model 'gen3_lite_gen3_lite_2f'...
# [ INFO] [1731979805.250983755]: No root/virtual joint specified in SRDF. Assuming fixed joint
# [ WARN] [1731979806.105148053, 673.289000000]: Could not identify parent group for end-effector 'end_effector'
# [ INFO] [1731979807.627123562, 673.619000000]: Ready to take commands for planning group arm.
# [ INFO] [1731979809.380881384, 674.280000000]: Ready to take commands for planning group gripper.
# [INFO] [1731979809.383121, 674.280000]: Initializing node in namespace /my_gen3_lite/
# [INFO] [1731979809.388017, 674.281000]: Reaching Named Target Home...
# [INFO] [1731979809.391064, 674.281000]: Going to named target home
# True

# Give your environment for your Kinova arm!
# ---------------------------
# Describe your workspace and the objects present.

# ChatGPT will then help control the robot.

# Currently, you need to supply the poses

# Of the objects in the form:

# position.x = num, position.y = num, position.z = num, orientation.x = num, orientation.y = num, orientation.z = num, orientation.w = num

# Grasp pose detection coming soon!

# commands
# ---------------------------
# enter : sends your input to ChatGPT

# CTRL-C to quit

# Your object detection system found the following objects in the scene: 1 person, 1 bottle, 2 bowls, 1 sandwich, 2 chairs, 2 dining tables. Which one of these objects is most likely to be trash? Please only put quotation marks around the beginning and end of your response.
# THIS IS YOUR INPUT:
# Your object detection system found the following objects in the scene: 1 person, 1 bottle, 2 bowls, 1 sandwich, 2 chairs, 2 dining tables. Which one of these objects is most likely to be trash? Please only put quotation marks around the beginning and end of your response.
# [INFO] [1731979823.510847, 677.798000]: Successfully got response from GPT-4!
# [INFO] [1731979823.513361, 677.798000]: Response from GPT-4:
# [INFO] [1731979823.516829, 677.798000]: "bottle"
# bottle

# Give your environment for your Kinova arm!
# ---------------------------
# Describe your workspace and the objects present.

# ChatGPT will then help control the robot.

# Currently, you need to supply the poses

# Of the objects in the form:

# position.x = num, position.y = num, position.z = num, orientation.x = num, orientation.y = num, orientation.z = num, orientation.w = num

# Grasp pose detection coming soon!

# commands
# ---------------------------
# enter : sends your input to ChatGPT

# CTRL-C to quit

# Now, provide code to pick up the trash. Example output: ['sceneObjects["gripper"]["open"]', 'sceneObjects["trash_object"]["go_to"]', 'sceneObjects["gripper"]["close"]', 'sceneObjects["trash_bin"]["go_to"]', 'sceneObjects["gripper"]["open"]', 'sceneObjects["home"]["go_to"]']. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable. You correctly identified the bottle as the trash object, but it actually a can.
# THIS IS YOUR INPUT:
# Now, provide code to pick up the trash. Example output: ['sceneObjects["gripper"]["open"]', 'sceneObjects["trash_object"]["go_to"]', 'sceneObjects["gripper"]["close"]', 'sceneObjects["trash_bin"]["go_to"]', 'sceneObjects["gripper"]["open"]', 'sceneObjects["home"]["go_to"]']. Also, I'm going to execute your code in Python, so please do not explain or provide any additional text, just provide the list. You don't need to assign the list to a variable. You correctly identified the bottle as the trash object, but it actually a can.
# [INFO] [1731979852.862527, 685.753000]: Successfully got response from GPT-4!
# [INFO] [1731979852.865104, 685.753000]: Response from GPT-4:
# [INFO] [1731979852.868847, 685.753000]: ['sceneObjects["gripper"]["open"]',
#  'sceneObjects["can"]["go_to"]',
#  'sceneObjects["gripper"]["close"]',
#  'sceneObjects["trash_bin"]["go_to"]',
#  'sceneObjects["gripper"]["open"]',
#  'sceneObjects["home"]["go_to"]']
# [INFO] [1731979852.870790, 685.753000]: Performing action...
# [INFO] [1731979853.051889, 685.795000]: Actual cartesian pose is : 
# [INFO] [1731979853.056086, 685.795000]: position: 
#   x: 0.4388089372744062
#   y: 0.19350585720364832
#   z: 0.4487533835696663
# orientation: 
#   x: 0.18928074392257485
#   y: 0.6846298451550569
#   z: 0.6813167295935048
#   w: 0.17681145064472326
# [INFO] [1731979853.060800, 685.795000]: do gripper stuff...
# [ INFO] [1731979854.810374904, 686.481000000]: Ready to take commands for planning group gripper.
# True
# [INFO] [1731979855.113977, 686.596000]: Performing action...
# [INFO] [1731979855.173661, 686.616000]: Actual cartesian pose is : 
# [INFO] [1731979855.175910, 686.616000]: position: 
#   x: 0.43880889770264186
#   y: 0.1935057487313796
#   z: 0.4487530088532156
# orientation: 
#   x: 0.18927981852033246
#   y: 0.684630655431633
#   z: 0.681315915793532
#   w: 0.17681243969411206
# [INFO] [1731979855.178066, 686.616000]: do object grabbing stuff...
# [INFO] [1731979855.180143, 686.616000]: Planning and going to the Cartesian Pose
# True
# [INFO] [1731979881.861064, 695.136000]: Actual cartesian pose is : 
# [INFO] [1731979881.862735, 695.136000]: position: 
#   x: -0.002671859662191964
#   y: -0.45822850055266245
#   z: 0.033353134405214535
# orientation: 
#   x: 0.053465279337720543
#   y: -0.7760368259676832
#   z: 0.6280533244612191
#   w: 0.02138528186754303
# [INFO] [1731979881.866265, 695.136000]: Planning and going to the Cartesian Pose
# True
# [INFO] [1731979899.926410, 698.186000]: Performing action...
# [INFO] [1731979899.955151, 698.195000]: Actual cartesian pose is : 
# [INFO] [1731979899.959063, 698.195000]: position: 
#   x: 0.008276035437670245
#   y: -0.6097191778599691
#   z: 0.026184573443990736
# orientation: 
#   x: 0.04664769715903683
#   y: -0.7691334057546113
#   z: 0.6373383841323332
#   w: 0.007588189089519622
# [INFO] [1731979899.964916, 698.195000]: do gripper stuff...
# True
# [INFO] [1731979906.232007, 699.536000]: Performing action...
# [INFO] [1731979906.315406, 699.555000]: Actual cartesian pose is : 
# [INFO] [1731979906.320224, 699.556000]: position: 
#   x: 0.008256957392203083
#   y: -0.6097192747909304
#   z: 0.026185608635725086
# orientation: 
#   x: 0.04663943192059548
#   y: -0.7691332393121797
#   z: 0.6373390664783196
#   w: 0.0075985469546412275
# [INFO] [1731979906.322784, 699.556000]: do trash can stuff...
# [INFO] [1731979906.325984, 699.556000]: Planning and going to the Cartesian Pose
# True
# [INFO] [1731979934.156682, 706.006000]: Performing action...
# [INFO] [1731979934.410412, 706.015000]: Actual cartesian pose is : 
# [INFO] [1731979934.413259, 706.015000]: position: 
#   x: -0.008759140931136177
#   y: 0.46253149355018547
#   z: 0.19245008734484853
# orientation: 
#   x: 0.0819849944717199
#   y: 0.6676889678080357
#   z: 0.7399051587965684
#   w: 0.003202957248746004
# [INFO] [1731979934.416896, 706.016000]: do gripper stuff...
# True
# [INFO] [1731979939.693409, 707.406000]: Performing action...
# [INFO] [1731979939.718139, 707.416000]: Actual cartesian pose is : 
# [INFO] [1731979939.722699, 707.417000]: position: 
#   x: -0.008761440615293787
#   y: 0.4625325708676958
#   z: 0.19245039424841714
# orientation: 
#   x: 0.08198517340438938
#   y: 0.6676885932338483
#   z: 0.7399055175137611
#   w: 0.0031935810543456352
# [INFO] [1731979939.728190, 707.417000]: do home stuff...
# [INFO] [1731979939.731607, 707.418000]: Going to named target home
# True



###################################################################################
# HUMAN ROBOT INTERACTION DEMO #
###################################################################################
# [INFO] [1734897937.622510, 81.517000]: Successfully got response from GPT-4!
# [INFO] [1734897937.623883, 81.517000]: Response from GPT-4:
# [INFO] [1734897937.624863, 81.517000]: {"most_likely_object": "canA", "certainty": "unsure", "explanation": "Without additional context such as the state or location of the cans, it's difficult to determine which is more likely to be trash. However, for the purpose of this task, I'll select canA as the most likely object."}
# {'most_likely_object': 'canA', 'certainty': 'unsure', 'explanation': "Without additional context such as the state or location of the cans, it's difficult to determine which is more likely to be trash. However, for the purpose of this task, I'll select canA as the most likely object."}

# Give your environment for your Kinova arm!
# ---------------------------
# Describe your workspace and the objects present.

# ChatGPT will then help control the robot.

# Currently, you need to supply the poses

# Of the objects in the form:

# position.x = num, position.y = num, position.z = num, orientation.x = num, orientation.y = num, orientation.z = num, orientation.w = num

# Grasp pose detection coming soon!

# commands
# ---------------------------
# enter : sends your input to ChatGPT

# CTRL-C to quit

# Can A is full, Can B is empty. Which one is most likely to be trash? Please format your response in the following way: {"most_likely_object": "object", "certainty": "sure_or_unsure", "explanation": "your_explanation"}
# THIS IS YOUR INPUT:
# Can A is full, Can B is empty. Which one is most likely to be trash? Please format your response in the following way: {"most_likely_object": "object", "certainty": "sure_or_unsure", "explanation": "your_explanation"}
# [INFO] [1734897979.979450, 90.531000]: Successfully got response from GPT-4!
# [INFO] [1734897980.031900, 90.531000]: Response from GPT-4:
# [INFO] [1734897980.039120, 90.531000]: {"most_likely_object": "Can B", "certainty": "sure", "explanation": "An empty can is more likely to be considered as trash because its purpose has been fulfilled."}

# Give your environment for your Kinova arm!
# ---------------------------
# Describe your workspace and the objects present.

# ChatGPT will then help control the robot.

# Currently, you need to supply the poses

# Of the objects in the form:

# position.x = num, position.y = num, position.z = num, orientation.x = num, orientation.y = num, orientation.z = num, orientation.w = num

# Grasp pose detection coming soon!

# commands
# ---------------------------
# enter : sends your input to ChatGPT

# CTRL-C to quit

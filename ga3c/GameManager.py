# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gym
from vizdoom import *
import numpy as np

class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        if game_name == "vizdoom":
            game = DoomGame()
            game.set_doom_scenario_path("basic.wad") #This corresponds to the simple task we will pose our agent
            game.set_doom_map("map01")
            game.set_screen_resolution(ScreenResolution.RES_160X120)
            game.set_screen_format(ScreenFormat.GRAY8)
            game.set_render_hud(False)
            game.set_render_crosshair(False)
            game.set_render_weapon(True)
            game.set_render_decals(False)
            game.set_render_particles(False)
            game.add_available_button(Button.MOVE_LEFT)
            game.add_available_button(Button.MOVE_RIGHT)
            game.add_available_button(Button.ATTACK)
            game.add_available_game_variable(GameVariable.AMMO2)
            game.add_available_game_variable(GameVariable.POSITION_X)
            game.add_available_game_variable(GameVariable.POSITION_Y)
            game.set_episode_timeout(300)
            game.set_episode_start_time(10)
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_living_reward(-1)
            game.set_mode(Mode.PLAYER)
            game.init()
            self.actions = self.actions = np.identity(len(game.get_available_buttons()),dtype=bool).tolist()
            #End Doom set-up
            self.env = game
        else:
            self.env = gym.make(game_name)
        self.display = display

        self.reset()

    def reset(self):
        self.env.new_episode()
        self.last_observation = self.env.get_state().screen_buffer
        return self.last_observation
        # observation = self.env.reset()
        # return observation

    def step(self, action):
        self._update_display()
        if self.game_name == "vizdoom":
            reward = self.env.make_action(self.actions[action]) / 100.0
            done = self.env.is_episode_finished()
            if not done:
                observation = self.env.get_state().screen_buffer
            else:
                observation = self.last_observation
            self.last_observation = observation
            info = None
            return observation, reward, done, info
        else:
            observation, reward, done, info = self.env.step(action)
            return observation, reward, done, info

    def num_actions(self):
        if (self.game_name == "vizdoom"):
            return len(self.env.get_available_buttons())

    def _update_display(self):
        if self.display:
            self.env.render()

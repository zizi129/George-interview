###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy
from copy import deepcopy

import queue
from queue import Queue
from threading import Thread, Event, Lock
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS,IndexTTS2,AzureTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def build_default_outline() -> dict:
    return {
        'summary': '',
        'focus_areas': [],
        'planned_questions': [],
        'depth_checks': [],
        'end_conditions': [],
    }


def build_default_outline_progress() -> dict:
    return {
        'decision': 'continue',
        'reason': '',
        'progress_summary': '',
        'covered_points': [],
        'remaining_points': [],
        'focus_prompt': '',
    }


def build_default_agent_state() -> dict:
    # Start at technical phase (index 1): the opening greeting is already sent
    # by interview_start, so the Agent's first job is technical questions.
    return {
        'current_phase': 'technical',
        'current_phase_idx': 1,
        'phase_turn_count': 0,
        'phase_main_question_count': 0,
        'current_card_id': '',
        'follow_up_budget': 0,
        'asked_card_ids': [],
        'phase_history': [],
        'evaluation_snapshot': {},
        'is_first_turn': True,
    }


def build_default_interview_context() -> dict:
    return {
        'job_title': '',
        'job_requirements': '',
        'interview_mode': 'text',
        'resume_file_name': '',
        'resume_profile': {},
        'interviewer_id': '',
        'interviewer_name': '',
        'interviewer_language': 'zh',
        'interviewer_language_label': '中文',
        'interview_outline': build_default_outline(),
        'outline_progress': build_default_outline_progress(),
        'agent_state': build_default_agent_state(),
        'interview_started_at': 0.0,
        'interview_finished': False,
        'interview_finish_reason': '',
        'interview_report_ready_at': 0.0,
        'wrap_up_turns': 0,
    }

def play_audio(quit_event,queue):        
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=8,
        output=True,
        output_device_index=1,
    )
    stream.start_stream()
    # while queue.qsize() <= 0:
    #     time.sleep(0.1)
    while not quit_event.is_set():
        stream.write(queue.get(block=True))
    stream.close()

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid
        self._interview_lock = Lock()
        self._generation_id = 0
        self._message_id = 0
        self._chat_history = []
        self._interview_history = []
        self._interview_context = build_default_interview_context()
        self._reset_outline_progress_review_state_locked()

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        elif opt.tts == "doubao":
            self.tts = DoubaoTTS(opt,self)
        elif opt.tts == "indextts2":
            self.tts = IndexTTS2(opt,self)
        elif opt.tts == "azuretts":
            self.tts = AzureTTS(opt,self)

        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

    def put_msg_txt(self,msg,datainfo:dict={}):
        self.tts.put_msg_txt(msg,datainfo)
    
    def put_audio_frame(self,audio_chunk,datainfo:dict={}): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,datainfo)

    def put_audio_file(self,filebyte,datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk],datainfo)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        with self._interview_lock:
            self._generation_id += 1
        self.tts.flush_talk()
        self.asr.flush_talk()

    def begin_generation(self) -> int:
        with self._interview_lock:
            self._generation_id += 1
            return self._generation_id

    def is_generation_current(self, generation_id: int) -> bool:
        with self._interview_lock:
            return self._generation_id == generation_id

    def _reset_outline_progress_review_state_locked(self):
        self._outline_review_requested_turn = 0
        self._outline_review_completed_turn = 0
        self._outline_review_completed_at = 0.0
        self._outline_review_model = ''
        self._outline_review_pending = False

    def request_outline_progress_review(self, user_turns: int) -> bool:
        try:
            target_turn = max(int(user_turns), 0)
        except (TypeError, ValueError):
            return False
        if target_turn <= 0:
            return False

        with self._interview_lock:
            if target_turn <= self._outline_review_requested_turn:
                return False
            if target_turn <= self._outline_review_completed_turn:
                return False
            self._outline_review_requested_turn = target_turn
            self._outline_review_pending = True
            return True

    def complete_outline_progress_review(self, user_turns: int, progress: dict, model_name: str = '') -> bool:
        try:
            target_turn = max(int(user_turns), 0)
        except (TypeError, ValueError):
            return False
        if target_turn <= 0 or not isinstance(progress, dict):
            return False

        with self._interview_lock:
            if target_turn != self._outline_review_requested_turn:
                return False
            next_context = dict(self._interview_context)
            next_context['outline_progress'] = deepcopy(progress)
            self._interview_context = next_context
            self._outline_review_completed_turn = target_turn
            self._outline_review_completed_at = time.time()
            self._outline_review_model = (model_name or '').strip()
            self._outline_review_pending = False
            return True

    def fail_outline_progress_review(self, user_turns: int):
        try:
            target_turn = max(int(user_turns), 0)
        except (TypeError, ValueError):
            return
        with self._interview_lock:
            if target_turn == self._outline_review_requested_turn:
                self._outline_review_pending = False

    def get_outline_progress_review_state(self) -> dict:
        with self._interview_lock:
            return {
                'pending': self._outline_review_pending,
                'requested_turn': self._outline_review_requested_turn,
                'completed_turn': self._outline_review_completed_turn,
                'completed_at': self._outline_review_completed_at,
                'model': self._outline_review_model,
            }

    def set_interview_context(
        self,
        job_title: str | None = None,
        job_requirements: str | None = None,
        interview_mode: str | None = None,
        resume_file_name: str | None = None,
        resume_profile: dict | None = None,
        interviewer_id: str | None = None,
        interviewer_name: str | None = None,
        interviewer_language: str | None = None,
        interviewer_language_label: str | None = None,
        interview_outline: dict | None = None,
        outline_progress: dict | None = None,
        agent_state: dict | None = None,
        interview_started_at: float | None = None,
        interview_finished: bool | None = None,
        interview_finish_reason: str | None = None,
        interview_report_ready_at: float | None = None,
        wrap_up_turns: int | None = None,
    ):
        with self._interview_lock:
            next_context = dict(self._interview_context)
            if job_title is not None:
                next_context['job_title'] = (job_title or '').strip()
            if job_requirements is not None:
                next_context['job_requirements'] = (job_requirements or '').strip()
            if interview_mode is not None:
                next_context['interview_mode'] = 'voice' if interview_mode == 'voice' else 'text'
            if resume_file_name is not None:
                next_context['resume_file_name'] = (resume_file_name or '').strip()
            if resume_profile is not None:
                next_context['resume_profile'] = dict(resume_profile) if isinstance(resume_profile, dict) else {}
            if interviewer_id is not None:
                next_context['interviewer_id'] = (interviewer_id or '').strip()
            if interviewer_name is not None:
                next_context['interviewer_name'] = (interviewer_name or '').strip()
            if interviewer_language is not None:
                next_context['interviewer_language'] = 'en' if interviewer_language == 'en' else 'zh'
            if interviewer_language_label is not None:
                next_context['interviewer_language_label'] = (interviewer_language_label or '').strip() or ('English' if next_context['interviewer_language'] == 'en' else '中文')
            if interview_outline is not None:
                next_context['interview_outline'] = deepcopy(interview_outline) if isinstance(interview_outline, dict) else build_default_outline()
            if outline_progress is not None:
                next_context['outline_progress'] = deepcopy(outline_progress) if isinstance(outline_progress, dict) else build_default_outline_progress()
            if agent_state is not None:
                next_context['agent_state'] = deepcopy(agent_state) if isinstance(agent_state, dict) else build_default_agent_state()
            if interview_started_at is not None:
                try:
                    next_context['interview_started_at'] = max(float(interview_started_at), 0.0)
                except (TypeError, ValueError):
                    next_context['interview_started_at'] = 0.0
            if interview_finished is not None:
                next_context['interview_finished'] = bool(interview_finished)
            if interview_finish_reason is not None:
                next_context['interview_finish_reason'] = (interview_finish_reason or '').strip()
            if interview_report_ready_at is not None:
                try:
                    next_context['interview_report_ready_at'] = max(float(interview_report_ready_at), 0.0)
                except (TypeError, ValueError):
                    next_context['interview_report_ready_at'] = 0.0
            if wrap_up_turns is not None:
                try:
                    next_context['wrap_up_turns'] = max(int(wrap_up_turns), 0)
                except (TypeError, ValueError):
                    next_context['wrap_up_turns'] = 0
            self._interview_context = next_context

    def get_interview_context(self) -> dict:
        with self._interview_lock:
            context = deepcopy(self._interview_context)
            started_at = float(context.get('interview_started_at') or 0.0)
            context['interview_elapsed_seconds'] = max(0.0, time.time() - started_at) if started_at > 0 else 0.0
            return context

    def get_interview_elapsed_seconds(self) -> float:
        with self._interview_lock:
            started_at = float(self._interview_context.get('interview_started_at') or 0.0)
        if started_at <= 0:
            return 0.0
        return max(0.0, time.time() - started_at)

    def increment_wrap_up_turns(self) -> int:
        with self._interview_lock:
            next_context = dict(self._interview_context)
            next_value = max(int(next_context.get('wrap_up_turns') or 0) + 1, 0)
            next_context['wrap_up_turns'] = next_value
            self._interview_context = next_context
            return next_value

    def mark_interview_finished(self, reason: str = '', report_ready_at: float = 0.0):
        with self._interview_lock:
            next_context = dict(self._interview_context)
            next_context['interview_finished'] = True
            next_context['interview_finish_reason'] = (reason or '').strip()
            try:
                next_context['interview_report_ready_at'] = max(float(report_ready_at), 0.0)
            except (TypeError, ValueError):
                next_context['interview_report_ready_at'] = 0.0
            self._interview_context = next_context

    def add_interview_message(self, role: str, content: str, metadata: dict | None = None):
        clean_content = (content or '').strip()
        if not clean_content:
            return None

        with self._interview_lock:
            self._message_id += 1
            entry = {
                'id': self._message_id,
                'role': 'assistant' if role == 'assistant' else 'user',
                'content': clean_content,
                'timestamp': round(time.time(), 3),
                'metadata': metadata or {},
            }
            self._chat_history.append(entry)
            self._chat_history = self._chat_history[-40:]
            self._interview_history.append(entry)
            return dict(entry)

    def get_recent_history(self, limit: int = 24) -> list:
        with self._interview_lock:
            return [dict(item) for item in self._chat_history[-limit:]]

    def get_history_since(self, last_id: int = 0) -> list:
        with self._interview_lock:
            return [dict(item) for item in self._interview_history if item['id'] > last_id]

    def get_interview_history(self) -> list:
        with self._interview_lock:
            return [dict(item) for item in self._interview_history]

    def get_user_turn_count(self) -> int:
        with self._interview_lock:
            return sum(1 for item in self._interview_history if item['role'] == 'user')

    def get_interview_transcript_text(self) -> str:
        with self._interview_lock:
            lines = []
            for item in self._interview_history:
                speaker = '面试官' if item['role'] == 'assistant' else '候选人'
                lines.append(f"{speaker}: {item['content']}")
            return '\n'.join(lines)

    def reset_interview_state(self, keep_context: bool = True):
        self.flush_talk()
        with self._interview_lock:
            self._chat_history = []
            self._interview_history = []
            self._message_id = 0
            self._reset_outline_progress_review_state_locked()
            if not keep_context:
                self._interview_context = build_default_interview_context()
            else:
                next_context = dict(self._interview_context)
                next_context['interview_outline'] = build_default_outline()
                next_context['outline_progress'] = build_default_outline_progress()
                next_context['interview_started_at'] = 0.0
                next_context['interview_finished'] = False
                next_context['interview_finish_reason'] = ''
                next_context['interview_report_ready_at'] = 0.0
                next_context['wrap_up_turns'] = 0
                self._interview_context = next_context

    def get_interview_state(self, last_id: int = 0) -> dict:
        context = self.get_interview_context()
        return {
            'messages': self.get_history_since(last_id),
            'context': context,
            'speaking': self.is_speaking(),
            'sessionid': self.sessionid,
        }

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            if enable_transition:
                # 检测状态变化
                current_speaking = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "George Interview", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                self.record_audio_data(frame)
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1
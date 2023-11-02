import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb


if __name__ == '__main__':

    # 환경 생성 함수
    def make_env():
        def _init():
            env = gym.make('Hopper')
            return env
        return _init

    # 5개의 환경으로 벡터화된 환경 생성
    n_envs = 50
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # 평가 환경을 생성합니다. 평가는 일반적으로 하나의 환경에서 수행되지만 필요한 경우 여러 환경에서도 수행할 수 있습니다.
    eval_env = gym.make('Hopper')
    eval_env = DummyVecEnv([lambda: eval_env])

    is_train = True
    if is_train :    
        run = wandb.init(
            project="stable-baselines3",
            name="sb3_sac_hopper",
            sync_tensorboard=True,
        )

        # SAC 인스턴스를 생성합니다
        model = SAC("MlpPolicy", env, verbose=1, batch_size=32768, tensorboard_log='sac_hopper_tensorboard/')
        
        # 평가 콜백 추가
        eval_callback = EvalCallback(eval_env, log_path='eval/', n_eval_episodes=5, deterministic=True, eval_freq=50)
        
        # 학습을 수행합니다 (WandbCallback 및 EvalCallback 모두 사용)
        model.learn(total_timesteps=500000, callback=[WandbCallback(model_save_path='sac_hopper'), eval_callback])
        
        # 학습된 모델을 저장합니다
        model.save("sac_hopper")

    is_eval = False
    if is_eval :
        # 모델을 불러옵니다
        model = SAC.load("sac_hopper.zip")
        
        # 환경을 평가하고 결과를 출력합니다
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done.all():
                obs = env.reset()

        env.close()

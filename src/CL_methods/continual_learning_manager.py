import numpy as np
import torch
import logging
import gc
import time

from .cl_method_params import cl_method_params

from .agem import AGEM
from .gem import GEM
from .mir import MIR
from .der import DER
from .ser import SER
from .er import ER
from .upper_bound import UPPER_BOUND
from .lower_bound import LOWER_BOUND
from .finetuning import FINETUNING

from .EpisodeBalancedBuffer import EpisodeBalancedBuffer
from .gss_greedy import GSS_Greedy
from .latent_greedy import Latent_Greedy
from .loss_greedy import Loss_Greedy
from .variance_greedy import Variance_Greedy

class CL_Manager:
    def __init__(self,
                 dataset_manager,
                 model,
                 config,
                 ):
        cl_method_params(config)
        self.dataset_manager = dataset_manager
        self.config = config
        self.replay_buffer_save = [[], []]

        self.retrain_batch_idxs = sorted(dataset_manager.test[self.test_set_idx].retrain_batch_idxs(config.batches_per_exp))

        self.model = model
        self.total_time = 0.0

        self._initialize_method()


    def _initialize_method(self):
        if self.config.replay_strat in ['episode_balanced']:
            self.dataset_manager.replay_buffer = EpisodeBalancedBuffer(self.dataset_manager.replay_buffer)
        
        elif self.config.replay_strat in ['gss_greedy']:
            self.dataset_manager.replay_buffer = GSS_Greedy(self.dataset_manager.replay_buffer, self.model)
        
        elif self.config.replay_strat in ['latent_greedy']:
            self.dataset_manager.replay_buffer = Latent_Greedy(self.dataset_manager.replay_buffer, self.model)
        
        elif self.config.replay_strat in ['loss_greedy']:
            self.dataset_manager.replay_buffer = Loss_Greedy(self.dataset_manager.replay_buffer, self.model)
        
        elif self.config.replay_strat in ['variance_greedy']:
            self.dataset_manager.replay_buffer = Variance_Greedy(self.dataset_manager.replay_buffer, self.model)

        if self.config.cl_method in ["ser", "der"]:
            if self.config.cl_method == "der":
                self.dataset_manager.init_old_predictions(self.model)
            if self.config.cl_method == "ser":
                self.dataset_manager.init_old_predictions(self.model)
 
    def check_retraining(self, batch_idx):
        if batch_idx in self.retrain_batch_idxs:
            exp_idx = self.retrain_batch_idxs.index(batch_idx)
            logging.info(f"Updating replay buffer at batch {batch_idx} and experience {exp_idx}...")
            self.retraining()


    def save_for_replay(self, normalized_inputs: np.ndarray, anomaly_flags: np.ndarray):
        self.replay_buffer_save[0].append(normalized_inputs.copy())
        self.replay_buffer_save[1].append(anomaly_flags.copy())
        

    def retraining(self):
        logging.info(f"The number of raw data points is: {len(self.replay_buffer_save[0])*self.config.batch_size}")
        start_time = time.time()

        before_num_val = len(self.dataset_manager.val)

        # Save the current state of the replay buffer
        self.dataset_manager.extend_dataset(new_data = self.replay_buffer_save[0].copy(),
                                        anomaly_flag = self.replay_buffer_save[1].copy(), 
                                        validationsplit = self.config.validation_split_ratio,
                                        test_set_idx=self.test_set_idx)

        logging.info(f"increased validation size = {len(self.dataset_manager.val)-before_num_val}")
        logging.info(f"Indices in tmp = {len(self.dataset_manager.tmp_dataset)}")
        self.replay_buffer_save = [[], []]
        gc.collect()

        with torch.enable_grad():
            self.model.train()

            if self.config.cl_method in ['agem', 'gem', 'mir', 'er']:

                # setting possible new lr
                for param_group in self.model.optimizer.param_groups:
                    param_group['lr'] = self.config.lr*self.config.method_params.lr_multiplier

                # initializing the dataloaders
                replay_dataloader = self.dataset_manager.get_dataloader(dataset="replay", 
                                                                    batch_size=self.config.method_params.sample_size,
                                                                    shuffle=True)
                val_dataloader = self.dataset_manager.get_dataloader('val', 
                                                                self.config.val_batch_size, 
                                                                shuffle=False)
                exp_dataloader = self.dataset_manager.get_dataloader('tmp',
                                                                self.config.batch_size,
                                                                shuffle=True)
                
                if self.config.cl_method == 'agem':
                    logging.info("--- using strategy: agem ---")
                    strat = AGEM(Replay_dataloader=replay_dataloader, 
                                    device=self.config.device, 
                                    params=self.config.method_params)
                
                elif self.config.cl_method == 'gem':
                    logging.info("--- using strategy: gem ---")
                    strat = GEM(Replay_dataloader=replay_dataloader, 
                                device=self.config.device, 
                                params=self.config.method_params)
                    
                elif self.config.cl_method == 'mir':
                    logging.info("--- using strategy: mir ---")
                    strat = MIR(Replay_dataloader=replay_dataloader, 
                                device=self.config.device, 
                                params=self.config.method_params)
                
                elif self.config.cl_method == 'er':
                    logging.info("--- using strategy: er ---")
                    strat = ER(Replay_dataloader=replay_dataloader, 
                                device=self.config.device, 
                                params=self.config.method_params)

                trained_epochs, samples_per_epoch = strat.train(exp_dataloader=exp_dataloader,
                            eval_dataloader=val_dataloader,
                            model=self.model)
                torch.cuda.empty_cache()

            elif self.config.cl_method in ["ser", "der"]:
                self.dataset_manager.recalculate_exp_predictions(self.model)
                self.dataset_manager.tmp_dataset.output_old_preds = True if self.config.cl_method == "ser" else False

                # setting possible new lr
                for param_group in self.model.optimizer.param_groups:
                    param_group['lr'] = self.config.lr*self.config.method_params.lr_multiplier

                # initializing the dataloaders
                replay_dataloader = self.dataset_manager.get_dataloader(dataset="replay", 
                                                                    batch_size=self.config.method_params.sample_size,
                                                                    shuffle=True)
                val_dataloader = self.dataset_manager.get_dataloader('val', 
                                                                self.config.val_batch_size, 
                                                                shuffle=False)
                exp_dataloader = self.dataset_manager.get_dataloader('tmp',
                                                                self.config.method_params.tmp_sample_size,
                                                                shuffle=True)
                
                if self.config.cl_method == 'der':
                    logging.info("--- using strategy: der ---")
                    strat = DER(replay_dataloader=replay_dataloader, 
                                    device=self.config.device, 
                                    params=self.config.method_params)
                    
                elif self.config.cl_method == 'ser':
                    logging.info("--- using strategy: ser ---")
                    strat = SER(replay_dataloader=replay_dataloader, 
                                    device=self.config.device, 
                                    params=self.config.method_params)                

                trained_epochs, samples_per_epoch = strat.train(exp_dataloader=exp_dataloader,
                            eval_dataloader=val_dataloader,
                            model=self.model)
                torch.cuda.empty_cache()

            elif self.config.cl_method in ["upper_bound"]:
                # Accumulating all data in the replay buffer
                self.dataset_manager.replay_buffer.add_processed_data(self.dataset_manager.tmp_dataset.data_segments, 
                                                            self.dataset_manager.tmp_dataset.indices,
                                                            self.dataset_manager.tmp_dataset.old_preds)
                self.dataset_manager.reset_tmp()
                gc.collect()

                # initializing the dataloaders
                replay_dataloader = self.dataset_manager.get_dataloader(dataset="replay", 
                                                                    batch_size=self.config.method_params.sample_size,
                                                                    shuffle=True)
                val_dataloader = self.dataset_manager.get_dataloader('val', 
                                                                self.config.val_batch_size, 
                                                                shuffle=False)

                logging.info("--- using strategy: upper_bound ---")
                strat = UPPER_BOUND(replay_dataloader=replay_dataloader, 
                                device=self.config.device, 
                                params=self.config.method_params)                

                trained_epochs, samples_per_epoch = strat.train(eval_dataloader=val_dataloader,
                            model=self.model)
                torch.cuda.empty_cache()

            elif self.config.cl_method in ["lower_bound", "finetuning"]:
                ### Lower Bound: trains new model with the new data
                ### Finetuning: Finetunes old model only with new data

                # initializing the dataloaders
                val_dataloader = self.dataset_manager.get_dataloader('val', 
                                                                self.config.val_batch_size, 
                                                                shuffle=False)
                exp_dataloader = self.dataset_manager.get_dataloader('tmp',
                                                                self.config.method_params.tmp_sample_size,
                                                                shuffle=True)
                
                if self.config.cl_method == "finetuning":
                    logging.info("--- using strategy: finetuning ---")
                    strat = FINETUNING(exp_dataloader=exp_dataloader, 
                                    device=self.config.device, 
                                    params=self.config.method_params)  

                elif self.config.cl_method == "lower_bound":
                    logging.info("--- using strategy: lower_bound ---")
                    strat = LOWER_BOUND(exp_dataloader=exp_dataloader, 
                                    device=self.config.device, 
                                    params=self.config.method_params)                

                trained_epochs, samples_per_epoch = strat.train(eval_dataloader=val_dataloader,
                            model=self.model)
                torch.cuda.empty_cache()

            else: raise ValueError(f"Continual learning method: {self.config.cl_method} is not implemented. Please use continual_learning = False")

        self.total_time += time.time() - start_time

        if self.config.cl_method not in ["upper_bound"]: # for other methods, already done within the training.
            self.dataset_manager.replay_buffer.add_processed_data(self.dataset_manager.tmp_dataset.data_segments, 
                                                                    self.dataset_manager.tmp_dataset.indices,
                                                                    self.dataset_manager.tmp_dataset.old_preds)
            self.dataset_manager.reset_tmp()
        gc.collect()
        self.model.eval()
        return trained_epochs*samples_per_epoch
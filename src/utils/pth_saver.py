import os
import torch
import torch.nn as nn

class PthSaverLoss():
    def __init__(self, 
                 pth_dir: str, 
                 model_name: str,
                 save_disk: bool = True,
                 ) -> None:
        """init of the PthSaverLoss

        Args:
            pth_dir (str): pth dir
            model_name (str): model name
            save_disk (bool, optional): save disk mode (only one pth at the end of the training process). Defaults to True.
        """
        self.pth_dir = pth_dir
        self.save_disk = save_disk
        self.model_name = model_name
        self.best_loss = 100000
        self.best_model = None
        os.makedirs(self.pth_dir, exist_ok=True)
        
    def save(self, 
             model: nn.Module, 
             epoch: int, 
             loss: float, 
             ):
        """saves pth model; if save_disk is enabled, it saves based on loss value

        Args:
            model (nn.Module): model
            epoch (int): epoch
            loss (float): loss value
        """
        filename = f"{self.model_name}_epoch_{epoch}_loss_{loss:.4f}.pth"
        model_path = os.path.join(self.pth_dir, filename)
        if self.save_disk:
            if loss < self.best_loss:
                if self.best_model is not None:
                    print(f"Removing previous best model {self.best_model}")
                    os.remove(self.best_model)
                
                torch.save(model.state_dict(), model_path)
                print(f"Saved pth model at {model_path}.")
                # updating best fields
                self.best_model = model_path
                self.best_loss = loss
        else:
            torch.save(model.state_dict(), model_path)

class PthSaverAccuracy():
    
    def __init__(self, 
                 pth_dir: str, 
                 model_name: str,
                 save_disk: bool = True,
                 ) -> None:
        """init of the PthSaverAccuracy

        Args:
            pth_dir (str): pth dir
            model_name (str): model name
            save_disk (bool, optional): save disk mode (only one pth at the end of the training process). Defaults to True.
        """
        self.pth_dir = pth_dir
        self.save_disk = save_disk
        self.model_name = model_name
        self.best_acc = 0
        self.best_model = None
        
        os.makedirs(self.pth_dir, exist_ok=True)
        
        
    def save(self, 
             model: nn.Module, 
             epoch: int, 
             loss: float, 
             acc: float
             ):
        """saves pth model; if save_disk is enabled, it saves based on loss value

        Args:
            model (nn.Module): model
            epoch (int): epoch
            loss (float): loss value
            acc (float): acc value
        """
        
        filename = f"{self.model_name}_epoch_{epoch}_loss_{loss:.4f}_acc_{acc:.4f}.pth"
        model_path = os.path.join(self.pth_dir, filename)
        if self.save_disk:
            if acc > self.best_acc:
                if self.best_model is not None:
                    print(f"Removing previous best model {self.best_model}")
                    os.remove(self.best_model)
                
                torch.save(model.state_dict(), model_path)
                print(f"Saved pth model at {model_path}.")
                # updating best fields
                self.best_model = model_path
                self.best_acc = acc
        else:
            torch.save(model.state_dict(), model_path)

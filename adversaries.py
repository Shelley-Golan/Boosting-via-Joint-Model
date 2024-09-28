import torch.nn as nn
# from autoattack.autopgd_base import APGDAttack
from torch.distributions import laplace
from torch.distributions import uniform
import torch
from torch.autograd import Variable
import numpy as np
from autoattack.autopgd_base import APGDAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class APGD():
    """
    APGD attack (from AutoAttack) (Croce et al, 2020).
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        ord (int): (optional) the order of maximum distortion (inf or 2).
    """

    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, ord=np.inf, seed=1):
        # assert loss_fn in ['ce', 'dlr'], 'Only loss_fn=ce or loss_fn=dlr are supported!'
        assert ord in [2, np.inf], 'Only ord=inf or ord=2 are supported!'

        norm = 'Linf' if ord == np.inf else 'L2'
        self.apgd = APGDAttack(predict, n_restarts=n_restarts, n_iter=nb_iter, verbose=False, eps=eps, norm=norm,
                               eot_iter=1, rho=.75, seed=seed, device=device)
        self.apgd.loss = 'ce' # loss_fn

    def perturb(self, x, y):
        # start = time()
        x_adv = self.apgd.perturb(x, y)
        # print(f'iter time {time() - start}')
        r_adv = x_adv - x
        return x_adv, r_adv


class LinfAPGDAttack(APGD):
    """
    APGD attack (from AutoAttack) with order=Linf.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """

    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, seed=1, **kwargs):
        ord = np.inf
        super(LinfAPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, n_restarts=n_restarts, eps=eps, nb_iter=nb_iter, ord=ord, seed=seed)


class L2APGDAttack(APGD):
    """
    APGD attack (from AutoAttack) with order=L2.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """

    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, seed=1, **kwargs):
        ord = 2
        super(L2APGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, n_restarts=n_restarts, eps=eps, nb_iter=nb_iter, ord=ord, seed=seed)


class Attack(object):
    """
    Abstract base class for all attack classes.
    Arguments:
        predict (nn.Module): forward pass function.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    """

    def __init__(self, predict, clip_min, clip_max):
        self.predict = predict
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """
        Virtual method for generating the adversarial examples.
        Arguments:
            x (torch.Tensor): the model's input tensor.
            **kwargs: optional parameters used by child classes.
        Returns:
            adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _verify_and_process_inputs(self, x, y):

        x = replicate_input(x)
        if y is not None:
            y = replicate_input(y)
        return x, y

def replicate_input(x):
    """
    Clone the input tensor x.
    """
    return x.detach().clone()


def replicate_input_withgrad(x):
    """
    Clone the input tensor x and set requires_grad=True.
    """
    return x.detach().clone().requires_grad_()


def calc_l2distsq(x, y):
    """
    Calculate L2 distance between tensors x and y.
    """
    d = (x - y)**2
    return d.view(d.shape[0], -1).sum(dim=1)


def clamp(input, min=None, max=None):
    """
    Clamp a tensor by its minimun and maximun values.
    """
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following.
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following.
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    """
    Multpliy a batch of tensors with a float or vector.
    """
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def batch_clamp(float_or_vector, tensor):
    """
    Clamp a batch of tensors.
    """
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    """
    Returns the Lp norm of batch x.
    """
    batch_size = x.size(0)
    return x.abs().pow(p).reshape(batch_size, -1).sum(dim=1).pow(1. / p)


def _thresh_by_magnitude(theta, x):
    """
    Threshold by magnitude.
    """
    return torch.relu(torch.abs(x) - theta) * x.sign()


def clamp_by_pnorm(x, p, r):
    """
    Clamp tensor by its norm.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def is_float_or_torch_tensor(x):
    """
    Return whether input x is a float or a torch.Tensor.
    """
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    Arguments:
        x (torch.Tensor): tensor containing the gradients on the input.
        p (int): (optional) order of the norm for the normalization (1 or 2).
        small_constant (float): (optional) to avoid dividing by zero.
    Returns:
        normalized gradients.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    """
    Randomly initialize the perturbation.
    """
    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss



def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn, delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, early_stop=False, ref_batch=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for iterative attacks.
    Arguments:
        xvar (torch.Tensor): input data.
        yvar (torch.Tensor): input labels.
        predict (nn.Module): forward pass function.
        nb_iter (int): number of iterations.
        eps (float): maximum distortion.
        eps_iter (float): attack step size.
        loss_fn (nn.Module): loss function.
        delta_init (torch.Tensor): (optional) tensor contains the random initialization.
        minimize (bool): (optional) whether to minimize or maximize the loss.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    Returns:
        torch.Tensor containing the perturbed input,
        torch.Tensor containing the perturbation
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)
    delta = delta.to(xvar.dtype)

    stopped = torch.zeros(xvar.size(0), dtype=torch.bool, device=xvar.device)

    delta.requires_grad_()
    for ii in range(nb_iter):
        if stopped.all():
            break  # Stop if all images have met the early stop condition
        outputs = predict(xvar + delta)
        if early_stop:
            outputs_ref = predict(ref_batch)
            probs_real = torch.sigmoid(outputs_ref).max(dim=1).values
            probs = torch.sigmoid(outputs)
            stop_criterion = probs.max(dim=1).values > probs_real
            stopped |= stop_criterion  # Update stopped status

        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss
        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)

            stopped_expanded = stopped.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            update_mask = ~stopped_expanded.expand_as(grad)

            if early_stop:
                grad_selected = torch.zeros_like(grad)
                grad_selected[update_mask] = grad[update_mask]
                grad = grad_selected

            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord=inf and ord=2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    r_adv = x_adv - xvar
    return x_adv, r_adv


class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, rand_init_type='uniform', early_stop=False):
        super(PGDAttack, self).__init__(predict, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.rand_init_type = rand_init_type
        self.ord = ord
        self.targeted = targeted
        self.loss_fn = loss_fn # nn.CrossEntropyLoss(reduction="sum")
        self.early_stop = False
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None, ref_batch=None):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted
                labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns:
            torch.Tensor containing perturbed inputs,
            torch.Tensor containing the perturbation
        """
        # start = time()
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            if self.rand_init_type == 'uniform':
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            elif self.rand_init_type == 'normal':
                delta.data = 0.001 * torch.randn_like(x)  # initialize as in TRADES
            else:
                raise NotImplementedError(
                    'Only rand_init_type=normal and rand_init_type=uniform have been implemented.')

        x_adv, r_adv = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter, eps=self.eps, eps_iter=self.eps_iter, loss_fn=self.loss_fn,
            minimize=self.targeted, ord=self.ord, clip_min=self.clip_min, clip_max=self.clip_max, delta_init=delta,
            early_stop=self.early_stop, ref_batch=ref_batch
        )
        # print(f'iter time {time() - start}')
        return x_adv.data, r_adv.data


class LinfPGDAttack(PGDAttack):
    """
    PGD Attack with order=Linf
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type)


class L2PGDAttack(PGDAttack):
    """
    PGD Attack with order=L2
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform',  early_stop=False):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type, early_stop=early_stop)
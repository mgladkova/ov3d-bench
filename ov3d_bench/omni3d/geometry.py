"""Geometry and bookkeeping helpers used by the AV2 and ScanNet converters.

LICENCE: CC-BY-NC 4.0. Trimmed from Omni3D / Cube R-CNN's `utils.py`
(Copyright (c) Meta Platforms, Inc. and affiliates), see LICENSE.md in this
directory. Only the 13 definitions the converters actually need are kept, out of 45.

Two deliberate changes from upstream: detectron2's `BoxMode.convert` is replaced by
the inline box conversion it performed, so detectron2 is not a dependency; and the
unused remainder of the file is dropped.

Do NOT reimplement `get_cuboid_verts` or `estimate_truncation`. They produce the
`bbox2D_proj` and `truncation` fields of generated ground truth, so any drift
silently changes the annotations and breaks reproduction of published numbers.
"""
import functools
import json
import os

import numpy as np
import torch


def array_converter(to_torch=True,
                    apply_to=tuple(),
                    template_arg_name_=None,
                    recover=True):
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy ndarrays
    for middle calculation, then convert output to original data-type if
    `recover=True`.

    Args:
        to_torch (Bool, optional): Whether convert to PyTorch tensors
            for middle calculation. Defaults to True.
        apply_to (tuple[str], optional): The arguments to which we apply
            data-type conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template (
            return arrays should have the same dtype and device
            as the template). Defaults to None. If None, we will use the
            first argument in `apply_to` as the template argument.
        recover (Bool, optional): Whether or not recover the wrapped function
            outputs to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or
            when apply_to contains an arg which is not among all args,
            a ValueError will be raised. When the template argument or
            an argument to convert is a list or tuple, and cannot be
            converted to a NumPy array, a ValueError will be raised.
        TypeError: When the type of the template argument or
                an argument to convert does not belong to the above range,
                or the contents of such an list-or-tuple-type argument
                do not share the same data type, a TypeError is raised.

    Returns:
        (function): wrapped function.

    Example:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add()
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not '
                                     f'an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(
                            input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper

class ArrayConverter:
    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32,
                                 np.int64, np.uint8, np.uint16, np.uint32,
                                 np.uint64, np.float16, np.float32, np.float64)

    def __init__(self, template_array=None):
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array):
        """Set template array.

        Args:
            array (tuple | list | int | float | np.ndarray | torch.Tensor):
                Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print(f'The following list cannot be converted to'
                      f' a numpy array of supported dtype:\n{array}')
                raise
        elif isinstance(array, self.SUPPORTED_NON_ARRAY_TYPES):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f'Template type {self.array_type}'
                            f' is not supported.')

    def convert(self, input_array, target_type=None, target_array=None):
        """Convert input array to target data type.

        Args:
            input_array (tuple | list | np.ndarray |
                torch.Tensor | int | float ):
                Input array. Defaults to None.
            target_type (<class 'np.ndarray'> | <class 'torch.Tensor'>,
                optional):
                Type to which input array is converted. Defaults to None.
            target_array (np.ndarray | torch.Tensor, optional):
                Template array to which input array is converted.
                Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print(f'The input cannot be converted to'
                      f' a single-type numpy array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, \
            'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), \
                'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(
                    input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), \
                'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(
                    target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array):
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), \
            'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(
                input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array

def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()

def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation.
    Args:
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space
        faces: the vertex indices per face
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # make sure types are correct
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    w3d = box3d[:, 3].unsqueeze(1)
    h3d = box3d[:, 4].unsqueeze(1)
    l3d = box3d[:, 5].unsqueeze(1)

    '''
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    '''

    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))

    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
    verts[:, 0, [1, 2, 5, 6]] = l3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
    verts[:, 1, [2, 3, 6, 7]] = h3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
    verts[:, 2, [4, 5, 6, 7]] = w3d / 2

    if R is not None:

        # rotate
        verts = R @ verts
    
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ]).float().unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze()
        faces = faces.squeeze()

    return verts, faces.to(verts.device)

def get_cuboid_verts(K, box3d, R=None, view_R=None, view_T=None):

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)

    n = len(box3d)

    if len(K.shape) == 2:
        K = K.unsqueeze(0).repeat([n, 1, 1])

    corners_3d, _ = get_cuboid_verts_faces(box3d, R)
    if view_T is not None:
        corners_3d -= view_T.view(1, 1, 3)
    if view_R is not None:
        corners_3d = (view_R @ corners_3d[0].T).T.unsqueeze(0)
    if view_T is not None:
        corners_3d[:, :, -1] += view_T.view(1, 1, 3)[:, :, -1]*1.25

    # project to 2D
    corners_2d = K @ corners_3d.transpose(1, 2)
    corners_2d[:, :2, :] = corners_2d[:, :2, :] / corners_2d[:, 2, :].unsqueeze(1)
    corners_2d = corners_2d.transpose(1, 2)

    if squeeze:
        corners_3d = corners_3d.squeeze()
        corners_2d = corners_2d.squeeze()

    return corners_2d, corners_3d

def intersect(box_a, box_b, mode='cross'):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    data_type = type(box_a)

    # this mode computes the intersect in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))

def iou(box_a, box_b, mode='cross', ign_area_b=False):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (array or tensor): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (array or tensor): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        ign_area_b (bool): if true then we ignore area of b. e.g., checking % box a is inside b
    """

    data_type = type(box_a)

    # this mode computes the IoU in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0)
            if not ign_area_b:
                union = union + area_b.unsqueeze(1) - inter

            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) 
            if not ign_area_b:
                union = union + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))

def convert_3d_box_to_2d(K, box3d, R=None, clipw=0, cliph=0, XYWH=True, min_z=0.20):
    """
    Converts a 3D box to a 2D box via projection. 
    Args:
        K (np.array): intrinsics matrix 3x3
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
        clipw (int): clip invalid X to the image bounds. Image width is usually used here.
        cliph (int): clip invalid Y to the image bounds. Image height is usually used here.
        XYWH (bool): returns in XYWH if true, otherwise XYXY format. 
        min_z: the threshold for how close a vertex is allowed to be before being
            considered as invalid for projection purposes.
    Returns:
        box2d (flexible): the 2D box results.
        behind_camera (bool): whether the projection has any points behind the camera plane.
        fully_behind (bool): all points are behind the camera plane. 
    """

    # bounds used for vertices behind image plane
    topL_bound = torch.tensor([[0, 0, 0]]).float()
    topR_bound = torch.tensor([[clipw-1, 0, 0]]).float()
    botL_bound = torch.tensor([[0, cliph-1, 0]]).float()
    botR_bound = torch.tensor([[clipw-1, cliph-1, 0]]).float()

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)
    verts2d, verts3d = get_cuboid_verts(K, box3d, R)

    # any boxes behind camera plane?
    verts_behind = verts2d[:, :, 2] <= min_z
    behind_camera = verts_behind.any(1)

    verts_signs = torch.sign(verts3d)

    # check for any boxes projected behind image plane corners
    topL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] < 0)
    topR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] < 0)
    botL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] > 0)
    botR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] > 0)
    
    # clip values to be in bounds for invalid points
    verts2d[topL] = topL_bound
    verts2d[topR] = topR_bound
    verts2d[botL] = botL_bound
    verts2d[botR] = botR_bound

    x, xi = verts2d[:, :, 0].min(1)
    y, yi = verts2d[:, :, 1].min(1)
    x2, x2i = verts2d[:, :, 0].max(1)
    y2, y2i = verts2d[:, :, 1].max(1)

    fully_behind = verts_behind.all(1)

    width = x2 - x
    height = y2 - y

    if XYWH:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)), dim=1)
    else:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)

    if squeeze:
        box2d = box2d.squeeze()
        behind_camera = behind_camera.squeeze()
        fully_behind = fully_behind.squeeze()

    return box2d, behind_camera, fully_behind

def estimate_truncation(K, box3d, R, imW, imH):

    box2d, out_of_bounds, fully_behind =  convert_3d_box_to_2d(K, box3d, R, imW, imH)
    
    if fully_behind:
        return 1.0

    box2d = box2d.detach().cpu().numpy().tolist()
    # upstream used detectron2 BoxMode.convert(XYWH_ABS -> XYXY_ABS)
    _x, _y, _w, _h = box2d
    box2d_XYXY = [_x, _y, _x + _w, _y + _h]
    image_box = np.array([0, 0, imW-1, imH-1])

    truncation = 1 - iou(np.array(box2d_XYXY)[np.newaxis], image_box[np.newaxis], ign_area_b=True)

    return truncation.item()

def load_json(path):
    
    with open(path, 'r') as fp:
        data = json.load(fp)

    return data

def save_json(path, data):

    with open(path, 'w') as fp:
        json.dump(data, fp)

def get_global_dataset_stats(path_to_stats=None, reset=False):

    if path_to_stats is None:
        path_to_stats = os.path.join('annotations', 'stats.json')

    if os.path.exists(path_to_stats) and not reset:
        stats = load_json(path_to_stats)
    
    else:
        stats = {
            'n_datasets': 0,
            'n_ims': 0,
            'n_anns': 0,
            'category_names' : [],
            'categories' : []
        }

    return stats

def save_global_dataset_stats(stats, path_to_stats=None):

    if path_to_stats is None:
        path_to_stats = os.path.join('annotations', 'stats.json')

    save_json(path_to_stats, stats)


import cupy


def _pad_boundary_ops(mode, var_name, size, int_t="int"):
    T = 'int' if int_t == 'int' else 'long long'
    min_func = 'min'
    max_func = 'max'
    if mode == 'constant':
        ops = f'''
        if (({var_name} < 0) || {var_name} >= {size}) {{
            {var_name} = -1;
        }}'''
    elif mode == 'symmetric':
        ops = f'''
        if ({var_name} < 0) {{
            {var_name} = - 1 -{var_name};
        }}
        {var_name} %= {size} * 2;
        {var_name} = {min_func}({var_name}, 2 * {size} - 1 - {var_name});'''
    elif mode == 'reflect':
        ops = f'''
        if ({size} == 1) {{
            {var_name} = 0;
        }} else {{
            if ({var_name} < 0) {{
                {var_name} = -{var_name};
            }}
            {var_name} = 1 + ({var_name} - 1) % (({size} - 1) * 2);
            {var_name} = {min_func}({var_name}, 2 * {size} - 2 - {var_name});
        }}'''
    elif mode == 'edge':
        ops = f'''{var_name} = {min_func}({max_func}(static_cast<{T}>({var_name}), static_cast<{T}>(0)), static_cast<{T}>({size} - 1));'''  # noqa
    elif mode == 'wrap':
        ops = f'''
        {var_name} %= {size};
        if ({var_name} < 0) {{
            {var_name} += {size};
        }}'''
    return ops + "\n"


def _generate_size_vars(ndim, arr_name='arr', size_prefix='size', int_type='int', order='C'):
    """Store shape of a raw array into individual variables.

    Examples
    --------
    >>> print(_generate_size_vars(3, 'arr', 'size', 'int'))
    int size_0 = arr.shape()[0];
    int size_1 = arr.shape()[1];
    int size_2 = arr.shape()[2];
    """
    # TODO: This seems like a CuPy bug that shape() indices have to be reversed
    #       for the order='F' case!
    shape_range = range(ndim) if order=='C' else range(ndim - 1, -1, -1)
    set_size_vars = [f'{int_type} {size_prefix}_{i} = {arr_name}.shape()[{j}];'
                     for i, j in enumerate(shape_range)]
    return '\n'.join(set_size_vars) + '\n'


def _generate_indices_ops(
    ndim, size_prefix='size', int_type='int', index_prefix='ind', order='C',
):
    """Generate indices based existing variables.

    Assumes variables f'{size_prefix}_{i}' has the size along axis, i.

    Examples
    --------
    >>> print(_generate_indices_ops(3, 'size', 'int', 'ind'))
    int _i = i;
    int ind_2 = _i % size_2; _i /= size_2;
    int ind_1 = _i % size_1; _i /= size_1;
    int ind_0 = _i;
    """
    if order == 'C':
        _range = range(ndim - 1, 0, -1)
        idx_largest_stride = 0
    elif order =='F':
        _range = range(ndim - 1)
        idx_largest_stride = ndim - 1
    else:
        raise ValueError(f"Unknown order: {order}. Must be one of {'C', 'F'}.")
    body = [f'{int_type} {index_prefix}_{j} = _i % {size_prefix}_{j}; _i /= {size_prefix}_{j};'
            for j in _range]
    body = '\n'.join(body)
    return f'{int_type} _i = i;\n{body}\n{int_type} {index_prefix}_{idx_largest_stride} = _i;\n'


def _gen_raveled(ndim, shape_prefix='shape', index_prefix='i', order='C'):
    """Generate raveled index for c-ordered memory layout

    For index_prefix='i', the indices are (i_0, i_1, ....)
    For shape_prefix='i', the shape is (shape_0, shape_1, ....)
    """
    if ndim == 1:
        return f'{index_prefix}_0';

    # sort axes from largest to smallest stride
    if order == 'C':
        ax = tuple(range(ndim))
    elif order == 'F':
        ax = tuple(range(ndim - 1, -1, -1))
    else:
        raise ValueError(f"Unknown order: {order}. Must be one of {'C', 'F'}.")
    ops = f'({shape_prefix}_{ax[1]} * {index_prefix}_{ax[0]})'
    for j in range(1, ndim - 1):
        ops = f'({shape_prefix}_{ax[j + 1]} * ({index_prefix}_{ax[j]} + {ops}))'
    return f'{index_prefix}_{ax[ndim - 1]} + ' + ops


@cupy._util.memoize(for_each_device=True)
def _get_pad_kernel(ndim=3, int_type='int', mode='edge', cval=0.0, order='C'):
    # variables storing shape of the output array
    out_size_prefix = 'shape'
    operation = _generate_size_vars(ndim, arr_name='out', size_prefix=out_size_prefix, int_type=int_type, order=order)

    # variables storing shape of the input array
    in_size_prefix = 'ishape'
    operation += _generate_size_vars(ndim, arr_name='arr', size_prefix=in_size_prefix, int_type=int_type, order=order)

    # unraveled indices into the output array
    out_index_prefix = 'oi'
    operation += _generate_indices_ops(ndim, size_prefix=out_size_prefix, int_type=int_type, index_prefix=out_index_prefix, order=order)

    # compute unraveled indices into the input array
    # (i_0, i_1, ...)
    in_index_prefix = 'i'
    operation += '\n'.join([f'{int_type} {in_index_prefix}_{j} = {out_index_prefix}_{j} - pad_widths[{2*j}];' for j in range(ndim)])
    operation += '\n'
    input_indices = tuple(f'{in_index_prefix}_{j}' for j in range(ndim))

    # impose boundary condition
    if mode == "constant":
        for i, coord in enumerate(input_indices):
            operation += _pad_boundary_ops(mode, coord, f"{in_size_prefix}_{i}", int_type)
            operation += f"""
                if ({coord} == -1) {{
                    out[i] = static_cast<F>({cval});
                    return;
                }}
            """
    else:
        for i, coord in enumerate(input_indices):
            operation += _pad_boundary_ops(mode, coord, f"{in_size_prefix}_{i}", int_type)

    raveled_idx = _gen_raveled(ndim, shape_prefix=in_size_prefix, index_prefix=in_index_prefix, order=order)
    operation += f"""
    // set output based on raveled index into the input array
    // currently assumes C-ordered output
    out[i] = arr[{raveled_idx}];
    """

    kernel_name = f"pad_{ndim}d_order{order}_{mode}"
    if mode == "constant":
        kernel_name += f"_c{str(cval).replace('.', '_').replace('-', 'm')}"
    if int_type != "int":
        kernel_name += f"_{int_type.replace(' ', '_')}_idx"
    return cupy.ElementwiseKernel(
        in_params="raw F arr, raw I pad_widths",
        out_params="raw F out",
        operation=operation,
        name=kernel_name)

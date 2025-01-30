// components/ui/button.jsx
import React from 'react';

export function Button({
  className = '',
  variant = 'default',
  size = 'default',
  children,
  ...props
}) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none';

  const variants = {
    default: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border border-gray-300 bg-white hover:bg-gray-100',
    ghost: 'hover:bg-gray-100',
  };

  const sizes = {
    default: 'h-10 py-2 px-4',
    sm: 'h-8 px-3',
    lg: 'h-12 px-8',
  };

  const styles = `${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`;

  return (
    <button className={styles} {...props}>
      {children}
    </button>
  );
}

export default Button;
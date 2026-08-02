import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, Check, X } from 'lucide-react';

const SearchableCombobox = ({
  label,
  name,
  value = '',
  onChange,
  options = [],
  placeholder = "Type to search or enter custom value...",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState(value);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  // Sync searchTerm when external value prop changes
  useEffect(() => {
    setSearchTerm(value || '');
  }, [value]);

  // Filter options based on search query
  const filteredOptions = options.filter(opt =>
    opt.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
        setHighlightedIndex(-1);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e) => {
    const newVal = e.target.value;
    setSearchTerm(newVal);
    setIsOpen(true);
    setHighlightedIndex(-1);
    onChange({ target: { name, value: newVal } });
  };

  const handleSelectOption = (opt) => {
    setSearchTerm(opt);
    setIsOpen(false);
    setHighlightedIndex(-1);
    onChange({ target: { name, value: opt } });
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setSearchTerm('');
    onChange({ target: { name, value: '' } });
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex(prev => 
        prev < filteredOptions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex(prev => 
        prev > 0 ? prev - 1 : filteredOptions.length - 1
      );
    } else if (e.key === 'Enter') {
      if (highlightedIndex >= 0 && highlightedIndex < filteredOptions.length) {
        e.preventDefault();
        handleSelectOption(filteredOptions[highlightedIndex]);
      } else {
        setIsOpen(false);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
      setHighlightedIndex(-1);
    }
  };

  // Helper to highlight matching text query in options
  const renderHighlightedText = (text, query) => {
    if (!query) return text;
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, idx) => 
      part.toLowerCase() === query.toLowerCase() ? (
        <span key={idx} className="bg-[#D8C3A5]/60 font-bold text-[#3B2F2F]">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  return (
    <div className="space-y-1.5 relative" ref={containerRef}>
      {label && (
        <label className="block text-sm font-medium text-[#3B2F2F]">
          {label}
        </label>
      )}

      <div className="relative">
        <Search className="w-4 h-4 text-[#8B6B4A] absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
        
        <input
          ref={inputRef}
          type="text"
          name={name}
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full bg-[#F8F5F0] border border-[#E7DDD2] rounded-xl pl-10 pr-9 py-2.5 text-sm text-[#3B2F2F] placeholder-[#A67C52]/70 focus:outline-none focus:ring-2 focus:ring-[#8B6B4A] focus:bg-white transition-all shadow-xs"
        />

        {searchTerm ? (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-[#7A624A] hover:text-[#3B2F2F] rounded-md transition-all"
            title="Clear value"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        ) : (
          <ChevronDown 
            className={`w-4 h-4 text-[#7A624A] absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none transition-transform ${isOpen ? 'rotate-180' : ''}`} 
          />
        )}
      </div>

      {/* Floating Suggestions Dropdown */}
      {isOpen && (
        <div className="absolute left-0 right-0 top-full mt-1 bg-[#FFFDF9] border border-[#E7DDD2] rounded-xl shadow-lg z-50 max-h-56 overflow-y-auto divide-y divide-[#F4EFE6] text-xs">
          {filteredOptions.length > 0 ? (
            filteredOptions.map((opt, idx) => {
              const isSelected = opt.toLowerCase() === searchTerm.toLowerCase();
              const isHighlighted = idx === highlightedIndex;
              return (
                <div
                  key={idx}
                  onClick={() => handleSelectOption(opt)}
                  onMouseEnter={() => setHighlightedIndex(idx)}
                  className={`px-4 py-2.5 cursor-pointer flex items-center justify-between transition-colors ${
                    isHighlighted ? 'bg-[#F4EFE6] text-[#3B2F2F]' : 'hover:bg-[#F8F5F0] text-[#5C4A38]'
                  }`}
                >
                  <span className="truncate pr-2">
                    {renderHighlightedText(opt, searchTerm)}
                  </span>
                  {isSelected && <Check className="w-4 h-4 text-[#8B6B4A] flex-shrink-0" />}
                </div>
              );
            })
          ) : (
            <div className="px-4 py-3 text-xs text-[#7A624A] italic">
              No matching suggestions. Press enter or click outside to accept custom value: <strong className="text-[#3B2F2F]">"{searchTerm}"</strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchableCombobox;
